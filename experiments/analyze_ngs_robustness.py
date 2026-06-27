#!/usr/bin/env python
"""NGS Robustness Analysis: OOD Detection + Adversarial Attacks on MNIST"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

sys.path.insert(0, '/home/me/ngs')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
SAVE_PATH = '/home/me/ngs/results/ood_robustness.json'
MODEL_CKPT = '/home/me/ngs/results/ngs_mnist_robustness.pt'


class Flatten:
    def __call__(self, x):
        return x.view(-1)
    def __repr__(self):
        return self.__class__.__name__ + '()'

def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        Flatten(),
    ])
    mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST('/tmp/fashion_mnist', train=False, download=True, transform=transform)
    mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    mnist_test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    fashion_loader = DataLoader(fashion_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return mnist_train_loader, mnist_test_loader, fashion_loader


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def build_ngs_mnist():
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        k_init=8,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.DIRECT_ADAPTER,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.PRE_ALLOCATED,
        gamma_residual=0.1,
        tau=1.0,
        ema_decay=0.99,
    )
    return NGSModel(784, 10, config)


def train_ngs(model, train_loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss: {total_loss/total_samples:.4f}")
    return model


def eval_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# OOD metrics extractors
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_ood_scores(model, loader, max_batches=None):
    model.eval()
    max_confs = []
    entropies = []
    min_mahals = []

    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(DEVICE)
        out = model(x)
        logits = out.logits if hasattr(out, 'logits') else out
        probs = F.softmax(logits, dim=-1)

        # 1) Max softmax confidence
        max_conf = probs.max(dim=-1)[0]
        max_confs.append(max_conf.cpu().numpy())

        # 2) Routing entropy (from routing weight distribution)
        router = model.router
        if hasattr(router, 'active_mask'):
            active_idx = router.active_mask.nonzero(as_tuple=True)[0]
            mu = router.mu[active_idx]
            log_s = router.log_s[active_idx]

            z = model.p_down(x)  # [B, d]
            diff = z.unsqueeze(1) - mu.unsqueeze(0)   # [B, K, d]
            s_sq = torch.exp(2 * log_s) + 1e-6
            dist = ((diff ** 2) / s_sq).sum(dim=-1)   # [B, K]
            min_dist = dist.min(dim=-1)[0]            # [B]
            min_mahals.append(min_dist.cpu().numpy())

            # Ent routing. Here: per-sample weight distribution across active units
            # We'll compute the entropy over the Gaussian weights for each sample.
            # Compute log_w for each active unit -> softmax -> entropy
            log_alpha = router.log_alpha[active_idx]
            log_w = log_alpha.unsqueeze(0) - (0.5 / router.tau) * dist  # [B, K]
            p = F.softmax(log_w, dim=-1)  # [B, K]
            ent = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # [B]
            entropies.append(ent.cpu().numpy())
        else:
            min_mahals.append(np.zeros(x.size(0)))
            entropies.append(np.zeros(x.size(0)))

    max_confs = np.concatenate(max_confs)
    entropies = np.concatenate(entropies)
    min_mahals = np.concatenate(min_mahals)

    return {
        "max_confidence": max_confs,
        "routing_entropy": entropies,
        "min_mahal_dist": min_mahals,
    }


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------
def compute_auroc(ood_scores_in, ood_scores_out, reversed_sign=False):
    """Returns AUROC for OOD detection. Higher means better.
    If reversed_sign=True, lower score means more OOD (e.g., confidence)."""
    import sklearn.metrics
    labels = np.concatenate([np.ones(len(ood_scores_in)), np.zeros(len(ood_scores_out))])
    if not reversed_sign:
        scores = np.concatenate([ood_scores_in, ood_scores_out])
    else:
        scores = -np.concatenate([ood_scores_in, ood_scores_out])
    return sklearn.metrics.roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# PGD Attack
# ---------------------------------------------------------------------------
def pgd_attack(model, x, y, eps, alpha=0.01, num_iter=20):
    """PGD targeted attack. Returns perturbed x."""
    x_adv = x.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        out = model(x_adv)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        x_adv = x_adv + alpha * grad.sign()
        perturbation = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + perturbation, min=0, max=1).detach().requires_grad_(True)
    return x_adv.detach()


def eval_pgd_accuracy(model, loader, eps, max_batches=None):
    model.eval()
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y, eps=eps)
        out = model(x_adv)
        logits = out.logits if hasattr(out, 'logits') else out
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("NGS OOD Detection & Adversarial Robustness on MNIST")
    print("=" * 60)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    train_loader, mnist_test_loader, fashion_loader = get_mnist_loaders()

    # Build / load model
    if os.path.exists(MODEL_CKPT):
        print(f"Loading model from {MODEL_CKPT}")
        model = build_ngs_mnist()
        model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
    else:
        print("Training NGS on MNIST for 5 epochs ...")
        model = build_ngs_mnist()
        model = train_ngs(model, train_loader, epochs=EPOCHS, lr=LR)
        torch.save(model.state_dict(), MODEL_CKPT)

    model.to(DEVICE)
    model.eval()

    # Clean accuracy
    clean_acc = eval_accuracy(model, mnist_test_loader)
    print(f"Clean MNIST accuracy: {clean_acc*100:.2f}%")

    # OOD scores
    print("\nExtracting OOD scores ...")
    id_scores = extract_ood_scores(model, mnist_test_loader, max_batches=50)
    ood_scores = extract_ood_scores(model, fashion_loader, max_batches=50)

    auroc_conf = compute_auroc(id_scores["max_confidence"], ood_scores["max_confidence"], reversed_sign=True)
    auroc_ent  = compute_auroc(id_scores["routing_entropy"], ood_scores["routing_entropy"], reversed_sign=False)
    auroc_mahal = compute_auroc(id_scores["min_mahal_dist"], ood_scores["min_mahal_dist"], reversed_sign=False)

    print(f"AUROC (max softmax confidence): {auroc_conf:.4f}")
    print(f"AUROC (routing entropy):        {auroc_ent:.4f}")
    print(f"AUROC (min Mahalanobis dist):   {auroc_mahal:.4f}")

    # PGD attack robustness
    pgd_results = {}
    for eps in [0.01, 0.05, 0.1]:
        print(f"\nRunning PGD attack with eps={eps} ...")
        pgd_acc = eval_pgd_accuracy(model, mnist_test_loader, eps=eps, max_batches=20)
        pgd_results[f"eps_{eps:.2f}"] = {
            "accuracy": pgd_acc,
            "accuracy_drop": clean_acc - pgd_acc,
        }
        print(f"  PGD acc (eps={eps}): {pgd_acc*100:.2f}%  (drop: {(clean_acc - pgd_acc)*100:.2f}%")

    # Save results
    results = {
        "clean_accuracy": clean_acc,
        "ood_auroc": {
            "max_softmax_confidence": auroc_conf,
            "routing_entropy": auroc_ent,
            "min_mahalanobis_distance": auroc_mahal,
        },
        "pgd_robustness": pgd_results,
    }

    with open(SAVE_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {SAVE_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
