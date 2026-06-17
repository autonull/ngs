"""Online (single-pass) continual learning benchmark for NGS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from copy import deepcopy


def run_online_cl_benchmark(
    n_tasks: int = 10,
    epochs_per_task: int = 1,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./online_cl_results",
    latent_dim: int = 32,
    k_init: int = 32,
    max_k: int = 256,
    lr: float = 1e-3,
    batch_size: int = 64,
    replay_ratio: float = 0.5,
    kd_weight: float = 5.0,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running online CL ({n_tasks} tasks, {epochs_per_task} epoch each) on {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs
    from ngs.training import NGSTrainer, TrainerConfig
    from experiments.datasets import get_task_loaders, ReplayBuffer
    from experiments.metrics import evaluate_model_on_task

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.FACTORIZED_SUBSPACE,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.CONTINUOUS_DENSITY,
        memory_management=MemoryManagement.DYNAMIC_GROWTH,
        split_threshold=0.05,
        prune_threshold=0.01,
        num_subspaces=4,
        hypernetwork_code_dim=8,
        tau=1.0,
    )

    model = build_ngs(784, 10, config).to(device)

    trainer_config = TrainerConfig(
        lr=lr,
        epochs=epochs_per_task,
        batch_size=batch_size,
        replay_ratio=replay_ratio,
        kd_weight=kd_weight,
        adapt_every_epoch=True,
        split_thresh=0.05,
        prune_thresh=0.01,
        max_spawn_per_call=5,
    )
    trainer = NGSTrainer(model, trainer_config, device=device)

    replay_buffer = ReplayBuffer(max_size=5000)
    old_model = None

    acc_matrix = np.zeros((n_tasks, n_tasks))
    task_accs = []

    for task_id in range(n_tasks):
        train_loader, test_loader, _ = get_task_loaders(
            "split_mnist", task_id % 5, 2, batch_size
        )

        metrics = trainer.train_epoch(
            train_loader, replay_buffer=replay_buffer, old_model=old_model
        )

        for eval_task in range(task_id + 1):
            _, test_loader_t, _ = get_task_loaders(
                "split_mnist", eval_task % 5, 2, batch_size
            )
            acc = evaluate_model_on_task(model, test_loader_t, device)
            acc_matrix[task_id, eval_task] = acc

        for x, y in train_loader:
            x_flat = x.view(x.size(0), -1)
            y_onehot = F.one_hot(y, num_classes=10).float()
            replay_buffer.add(x_flat, y_onehot)

        old_model = deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        print(f"Task {task_id}: acc={acc_matrix[task_id, task_id]:.4f}, K={model.K}")

    accuracies = [acc_matrix[i, i] for i in range(n_tasks)]
    avg_acc = np.mean(accuracies)
    bwt = np.mean([acc_matrix[t, t] - acc_matrix[t - 1, t] for t in range(1, n_tasks)]) if n_tasks > 1 else 0.0
    fwt = np.mean([acc_matrix[0, t] - 0.5 for t in range(1, n_tasks)]) if n_tasks > 1 else 0.0

    results = {
        "n_tasks": n_tasks,
        "avg_acc": float(avg_acc),
        "bwt": float(bwt),
        "fwt": float(fwt),
        "accuracies": [float(a) for a in accuracies],
        "final_K": int(model.K),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"online_cl_{n_tasks}tasks.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / f'online_cl_{n_tasks}tasks.json'}")
    return results
