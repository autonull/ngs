"""Class-incremental learning benchmark for NGS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from copy import deepcopy


def compute_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            out_obj = model(x)
            logits = out_obj.logits
            _, pred = logits.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return correct / total


def run_class_incremental_benchmark(
    dataset: str = "cifar100",
    classes_per_task: int = 10,
    n_tasks: int = 10,
    epochs_per_task: int = 5,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./class_incremental_results",
    latent_dim: int = 64,
    k_init: int = 64,
    max_k: int = 512,
    lr: float = 1e-3,
    batch_size: int = 128,
    replay_memory: int = 2000,
    kd_weight: float = 5.0,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running class-incremental on {dataset} ({classes_per_task} classes/task, {n_tasks} tasks)")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset, TensorDataset

    if dataset == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        full_train = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
        full_test = datasets.CIFAR100("./data", train=False, download=True, transform=transform)
        n_classes = 100
        d_in = 3 * 32 * 32
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        full_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        full_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        n_classes = 10
        classes_per_task = min(classes_per_task, 5)
        n_tasks = min(n_tasks, n_classes // classes_per_task)
        d_in = 3 * 32 * 32
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs
    from ngs.training import NGSTrainer, TrainerConfig

    # Sort classes for incremental order
    all_classes = list(range(n_classes))
    task_classes = [all_classes[i:i + classes_per_task] for i in range(0, len(all_classes), classes_per_task)][:n_tasks]
    seen_classes = []

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        num_subspaces=max(4, min(8, n_classes // 10)),
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=32,
        tau=1.0,
    )

    model = build_ngs(d_in, n_classes, config).to(device)

    trainer_config = TrainerConfig(
        lr=lr,
        epochs=epochs_per_task,
        batch_size=batch_size,
        kd_weight=kd_weight,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
        max_spawn_per_call=5,
    )
    trainer = NGSTrainer(model, trainer_config, device=device)

    acc_matrix = np.zeros((n_tasks, n_tasks))
    replay_data, replay_labels = [], []
    old_model = None

    for task_id, classes in enumerate(task_classes):
        seen_classes.extend(classes)
        train_indices = [i for i, (_, y) in enumerate(full_train) if y in classes]
        test_indices = [i for i, (_, y) in enumerate(full_test) if y in classes]

        task_train = Subset(full_train, train_indices)
        task_test = Subset(full_test, test_indices)
        train_loader = DataLoader(task_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(task_test, batch_size=batch_size)

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                y_onehot = F.one_hot(y, num_classes=n_classes).float()

                optimizer = trainer.optimizer
                optimizer.zero_grad()
                out_obj = model(x)
                logits = out_obj.logits
                ce_loss = F.cross_entropy(logits, y)

                kd_loss = torch.tensor(0.0, device=device)
                if old_model is not None and kd_weight > 0:
                    with torch.no_grad():
                        old_out_obj = old_model(x)
                        old_logits = old_out_obj.logits
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / 2.0, dim=-1),
                        F.softmax(old_logits / 2.0, dim=-1),
                        reduction="batchmean",
                    ) * (2.0 ** 2)

                loss = ce_loss + kd_weight * kd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if hasattr(model, "adapt_density"):
                with torch.no_grad():
                    z_samples = []
                    for x, _ in train_loader:
                        x = x.view(x.size(0), -1).to(device)
                        z = model.p_down(x)
                        z_samples.append(z)
                        if len(torch.cat(z_samples)) >= 200:
                            break
                    if z_samples:
                        z_samples = torch.cat(z_samples)[:200]
                        model.adapt_density(
                            z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=5
                        )

        for eval_task in range(task_id + 1):
            eval_indices = [i for i, (_, y) in enumerate(full_test) if y in task_classes[eval_task]]
            eval_subset = Subset(full_test, eval_indices)
            eval_loader = DataLoader(eval_subset, batch_size=batch_size)
            acc = compute_accuracy(model, eval_loader, device)
            acc_matrix[task_id, eval_task] = acc

        old_model = deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        print(f"Task {task_id} classes={classes}: acc={acc_matrix[task_id, task_id]:.4f}, max_K={model.K}")

    avg_acc = np.mean([acc_matrix[i, i] for i in range(n_tasks)])
    if n_tasks > 1:
        bwt = np.mean([acc_matrix[t, t] - acc_matrix[t - 1, t] for t in range(1, n_tasks)])
    else:
        bwt = 0.0

    results = {
        "dataset": dataset,
        "n_tasks": n_tasks,
        "classes_per_task": classes_per_task,
        "avg_acc": float(avg_acc),
        "bwt": float(bwt),
        "acc_matrix": acc_matrix.tolist(),
        "final_K": int(model.K),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"class_incremental_{dataset}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / f'class_incremental_{dataset}.json'}")
    return results
