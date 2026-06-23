"""Meta-Gaussian controllers benchmark for NGS (Experiment 1D).
Tests self-referential growth: meta-NGS tunes split thresholds, subspaces online.
Target: 3-10x faster adaptation on domain shifts (PermutedMNIST)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
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


def run_metagaussian_benchmark(
    n_shifts: int = 5,
    epochs_per_shift: int = 5,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./metagaussian_results",
    latent_dim: int = 64,
    k_init: int = 64,
    max_k: int = 512,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running Meta-Gaussian controller benchmark: {n_shifts} domain shifts")

    from experiments.datasets import PermutedMNIST, get_transform
    from torch.utils.data import DataLoader
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs
    from ngs.training import NGSTrainer, TrainerConfig

    # Create permuted MNIST shifts
    permuted = PermutedMNIST(n_tasks=n_shifts, seed=seed)

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.META_LEARNED,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=32,
        tau=1.0,
        meta_lr=1e-3,
        meta_hidden_dim=64,
    )

    model = build_ngs(784, 10, config).to(device)

    trainer_config = TrainerConfig(
        lr=lr,
        epochs=epochs_per_shift,
        batch_size=batch_size,
        kd_weight=5.0,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
        max_spawn_per_call=10,
    )
    trainer = NGSTrainer(model, trainer_config, device=device)

    # Track adaptation speed
    shift_results = []
    old_model = None

    for shift_id in range(n_shifts):
        print(f"\n=== Shift {shift_id} ===")
        train_loader, test_loader = permuted.get_task_data(shift_id, batch_size)

        # Measure adaptation: accuracy at each epoch
        shift_accs = []

        for epoch in range(epochs_per_shift):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)

                optimizer = trainer.optimizer
                optimizer.zero_grad()
                out_obj = model(x)
                logits = out_obj.logits
                ce_loss = F.cross_entropy(logits, y)

                kd_loss = torch.tensor(0.0, device=device)
                if old_model is not None and trainer_config.kd_weight > 0:
                    with torch.no_grad():
                        old_out_obj = old_model(x)
                        old_logits = old_out_obj.logits
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / 2.0, dim=-1),
                        F.softmax(old_logits / 2.0, dim=-1),
                        reduction="batchmean",
                    ) * (2.0 ** 2)

                # Topology losses
                topo_losses = model.compute_topology_losses()
                loss = ce_loss + trainer_config.kd_weight * kd_loss
                loss += 0.01 * topo_losses.get('entropy', 0)
                loss += 0.01 * topo_losses.get('diversity', 0)
                if 'split_gate' in topo_losses:
                    loss += 0.1 * topo_losses['split_gate']

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
                            z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=10
                        )

            test_acc = compute_accuracy(model, test_loader, device)
            shift_accs.append(test_acc)
            print(f"  Epoch {epoch}: acc={test_acc:.4f}, K={model.K}")

        shift_results.append(shift_accs)

        # Save model for KD on next shift
        old_model = deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

    # Compute adaptation speed: epochs to reach 90% of final accuracy
    adaptation_speeds = []
    for accs in shift_results:
        final_acc = accs[-1]
        target = 0.9 * final_acc
        epochs_to_target = next((i for i, a in enumerate(accs) if a >= target), len(accs))
        adaptation_speeds.append(epochs_to_target)

    avg_final_acc = np.mean([s[-1] for s in shift_results])
    avg_adaptation = np.mean(adaptation_speeds)

    print(f"\n=== Summary ===")
    print(f"Avg final accuracy: {avg_final_acc:.4f}")
    print(f"Avg adaptation speed (epochs to 90%): {avg_adaptation:.2f}")
    print(f"Final K: {model.K}")

    results = {
        "n_shifts": n_shifts,
        "epochs_per_shift": epochs_per_shift,
        "shift_results": shift_results,
        "adaptation_speeds": adaptation_speeds,
        "avg_final_acc": float(avg_final_acc),
        "avg_adaptation_epochs": float(avg_adaptation),
        "final_K": int(model.K),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / f"metagaussian_{n_shifts}shifts.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir}")
    return results


# Compare with baseline (heuristic topology)
def run_baseline_comparison(
    n_shifts: int = 5,
    epochs_per_shift: int = 5,
    device: str = "cuda",
    seed: int = 42,
    latent_dim: int = 64,
    k_init: int = 64,
    max_k: int = 512,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Run same benchmark with heuristic topology for comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    from experiments.datasets import PermutedMNIST
    from torch.utils.data import DataLoader
    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs
    from ngs.training import NGSTrainer, TrainerConfig

    permuted = PermutedMNIST(n_tasks=n_shifts, seed=seed)

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC,
        split_threshold=0.05,
        prune_threshold=0.01,
        hypernetwork_code_dim=16,
        hypernetwork_hidden_dim=32,
        tau=1.0,
    )

    model = build_ngs(784, 10, config).to(device)
    trainer_config = TrainerConfig(
        lr=lr, epochs=epochs_per_shift, batch_size=batch_size, kd_weight=5.0,
        adapt_every_epoch=True, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=10
    )
    trainer = NGSTrainer(model, trainer_config, device=device)

    shift_results = []
    old_model = None

    for shift_id in range(n_shifts):
        train_loader, test_loader = permuted.get_task_data(shift_id, batch_size)
        shift_accs = []

        for epoch in range(epochs_per_shift):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                x = x.view(x.size(0), -1)
                optimizer = trainer.optimizer
                optimizer.zero_grad()
                out_obj = model(x)
                logits = out_obj.logits
                ce_loss = F.cross_entropy(logits, y)

                kd_loss = torch.tensor(0.0, device=device)
                if old_model is not None:
                    with torch.no_grad():
                        old_out_obj = old_model(x)
                        old_logits = old_out_obj.logits
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / 2.0, dim=-1),
                        F.softmax(old_logits / 2.0, dim=-1),
                        reduction="batchmean",
                    ) * 4.0

                loss = ce_loss + trainer_config.kd_weight * kd_loss
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
                            z_samples=z_samples, split_thresh=0.05, prune_thresh=0.01, max_spawn_per_call=10
                        )

            test_acc = compute_accuracy(model, test_loader, device)
            shift_accs.append(test_acc)

        shift_results.append(shift_accs)
        old_model = deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

    adaptation_speeds = []
    for accs in shift_results:
        final_acc = accs[-1]
        target = 0.9 * final_acc
        epochs_to_target = next((i for i, a in enumerate(accs) if a >= target), len(accs))
        adaptation_speeds.append(epochs_to_target)

    return {
        "shift_results": shift_results,
        "adaptation_speeds": adaptation_speeds,
        "avg_final_acc": float(np.mean([s[-1] for s in shift_results])),
        "avg_adaptation_epochs": float(np.mean(adaptation_speeds)),
        "final_K": int(model.K),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shifts", type=int, default=5)
    parser.add_argument("--epochs-per-shift", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()

    meta_results = run_metagaussian_benchmark(
        n_shifts=args.n_shifts,
        epochs_per_shift=args.epochs_per_shift,
        device=args.device,
        seed=args.seed,
    )

    if args.compare_baseline:
        print("\n=== Running Baseline Comparison ===")
        baseline_results = run_baseline_comparison(
            n_shifts=args.n_shifts,
            epochs_per_shift=args.epochs_per_shift,
            device=args.device,
            seed=args.seed,
        )
        print(f"Meta-Gaussian: {meta_results['avg_adaptation_epochs']:.2f} epochs to 90%")
        print(f"Baseline: {baseline_results['avg_adaptation_epochs']:.2f} epochs to 90%")
        speedup = baseline_results['avg_adaptation_epochs'] / meta_results['avg_adaptation_epochs']
        print(f"Speedup: {speedup:.2f}x")