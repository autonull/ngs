"""
Quick runner for standard benchmarks with pretrained backbones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import os
import json

from experiments.config import ExperimentConfig, EXPERIMENTS, TrainConfig, ModelConfig, as_train_kwargs
from experiments.datasets import get_task_loaders
from experiments.metrics import compute_metrics, evaluate_model_on_task
from experiments.backbones import create_backbone_ngs, PretrainedBackbone


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_backbone_ngs(model: nn.Module, train_loader: DataLoader, task_id: int,
                       old_model=None, device='cuda', epochs=5, lr=1e-3, 
                       weight_decay=1e-4, replay_buffer=None, replay_ratio=1.0,
                       kd_weight=2.0, kd_temperature=2.0, **kwargs):
    """Train only the NGS head (backbone frozen)."""
    model.to(device)
    model.train()
    
    # Only optimize head parameters
    optimizer = torch.optim.AdamW(model.ngs_head.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        model.ngs_head.train()
        losses = []
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            # Replay
            if replay_buffer and len(replay_buffer) > x.size(0):
                rx, ry = replay_buffer.sample(int(x.size(0) * replay_ratio))
                if rx is not None:
                    rx, ry = rx.to(device), ry.to(device)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry.argmax(dim=1)], dim=0)
            # Also move x, y to device if they're not already
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # CE loss
            ce_loss = F.cross_entropy(logits, y)
            
            # KD loss on replay samples
            kd_loss = 0
            if old_model is not None:
                with torch.no_grad():
                    old_logits = old_model(x)
                n_new = x.size(0) // (1 + int(replay_ratio))
                if n_new < x.size(0):
                    kd_loss = F.kl_div(
                        F.log_softmax(logits[n_new:] / kd_temperature, dim=-1),
                        F.softmax(old_logits[n_new:] / kd_temperature, dim=-1),
                        reduction='batchmean'
                    ) * (kd_temperature ** 2)
            
            total_loss = ce_loss + kd_weight * kd_loss
            total_loss.backward()
            model.ngs_head.update_grad_ema()
            optimizer.step()
            
            losses.append(ce_loss.item())
            
            # Update replay buffer
            if replay_buffer:
                with torch.no_grad():
                    replay_buffer.add(
                        x[:x.size(0)//2].detach().cpu(),
                        F.one_hot(y[:y.size(0)//2], num_classes=model.ngs_head.p_up.out_features).float().cpu()
                    )
        
        # Adaptive density control
        model.ngs_head.adapt_density(
            split_thresh=kwargs.get('split_thresh', 0.005),
            prune_thresh=kwargs.get('prune_thresh', 0.01),
            max_spawn_per_call=kwargs.get('max_spawn_per_call', 5)
        )
    
    return np.mean(losses), 0


def evaluate_model_on_task_4d(model, test_loader, device) -> float:
    """Evaluate model on a single task (4D input for backbone)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_quick_experiment(
    config: ExperimentConfig,
    backbone_name: str = 'resnet18',
    seed: int = 42,
    output_dir: str = './results',
    verbose: bool = True
) -> Dict:
    """Run experiment with pretrained backbone."""
    set_seed(seed)
    device = config.device if hasattr(config, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create backbone + NGS head
    model = create_backbone_ngs(
        backbone_name=backbone_name,
        num_classes=config.output_dim,
        d_latent=config.model.d_latent,
        k_init=config.model.k_init,
        max_k=config.model.max_k,
        top_k=config.model.top_k,
        freeze_backbone=True
    )
    
    train_kwargs = as_train_kwargs(config.train)
    
    # Setup replay buffer
    from experiments.datasets import ReplayBuffer
    replay_buffer = ReplayBuffer(max_size=config.train.replay_size)
    
    # Task loader function
    def get_task_data(task_id):
        train_loader, test_loader, classes = get_task_loaders(
            config.dataset, task_id, config.classes_per_task, config.train.batch_size
        )
        return train_loader, test_loader, classes
    
    # Run continual evaluation
    accuracy_matrix = np.zeros((config.n_tasks, config.n_tasks))
    active_units_list = []
    old_model = None
    
    for task_id in range(config.n_tasks):
        train_loader, test_loader, classes = get_task_data(task_id)
        
        # Train
        train_kwargs['replay_buffer'] = replay_buffer
        train_kwargs['old_model'] = old_model
        train_backbone_ngs(model, train_loader, task_id, device=device, **train_kwargs)
        
        # Evaluate on all seen tasks
        for eval_task in range(task_id + 1):
            _, eval_test_loader, _ = get_task_data(eval_task)
            acc = evaluate_model_on_task_4d(model, eval_test_loader, device)
            accuracy_matrix[eval_task, task_id] = acc
        
        # Update replay buffer (store raw images for backbone)
        for x, y in train_loader:
            replay_buffer.add(x, F.one_hot(y, num_classes=config.output_dim).float())
        
        # Save old model for KD
        old_model = create_backbone_ngs(
            backbone_name=backbone_name,
            num_classes=config.output_dim,
            d_latent=config.model.d_latent,
            k_init=config.model.k_init,
            max_k=config.model.max_k,
            top_k=config.model.top_k,
            freeze_backbone=True
        )
        old_model.to(device)
        # Copy trained head
        old_model.ngs_head.load_state_dict(model.ngs_head.state_dict())
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False
        
        # Track capacity
        if hasattr(model.ngs_head, 'active_mask'):
            active_units_list.append(model.ngs_head.active_mask.sum().item())
        else:
            active_units_list.append(0)
        
        if verbose:
            print(f"  Task {task_id} done. Acc on task {task_id}: {accuracy_matrix[task_id, task_id]:.4f}")
    
    # Compute metrics
    metrics = compute_metrics(accuracy_matrix, random_baseline=1.0 / config.output_dim)
    metrics.active_units = active_units_list[-1] if active_units_list else 0
    metrics.max_units = config.model.max_k
    
    # Count params
    head_params = sum(p.numel() for p in model.ngs_head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{config.name}_backbone{backbone_name}_ngs_seed{seed}.json")
    full_result = {
        'metrics': metrics.to_dict(),
        'accuracy_matrix': accuracy_matrix.tolist(),
        'active_units': active_units_list,
        'config': config.name,
        'model': 'ngs_backbone',
        'backbone': backbone_name,
        'head_params': head_params,
        'total_params': total_params,
        'seed': seed,
    }
    with open(result_file, 'w') as f:
        json.dump(full_result, f, indent=2)
    
    if verbose:
        from experiments.metrics import print_results
        print_results(metrics, f"{config.name} - NGS ({backbone_name}) seed={seed}")
        print(f"  Head params: {head_params:,}, Total params: {total_params:,}")
    
    return full_result


# Quick benchmark configs (standard CL benchmarks)
QUICK_BENCHMARKS = {
    'cifar100': ExperimentConfig(
        name='Split-CIFAR100',
        dataset='split_cifar100',
        scenario='class_incremental',
        n_tasks=10,
        classes_per_task=10,
        input_dim=32*32*3,
        output_dim=10,
        train=TrainConfig(
            epochs_per_task=5,
            batch_size=128,
            replay_size=20000,
            replay_ratio=1.0,
            kd_weight=2.0,
            kd_temperature=2.0,
            split_thresh=0.01,
            prune_thresh=0.01,
            max_spawn_per_call=5,
        ),
        model=ModelConfig(
            d_latent=64, k_init=32, max_k=256, top_k=8, lora_rank=4
        ),
    ),
    'cifar10': ExperimentConfig(
        name='Split-CIFAR10',
        dataset='split_cifar10',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=32*32*3,
        output_dim=2,
        train=TrainConfig(
            epochs_per_task=5,
            batch_size=128,
            replay_size=20000,
            replay_ratio=1.0,
            kd_weight=2.0,
            kd_temperature=2.0,
            split_thresh=0.01,
            prune_thresh=0.01,
            max_spawn_per_call=5,
        ),
        model=ModelConfig(
            d_latent=64, k_init=32, max_k=256, top_k=8, lora_rank=4
        ),
    ),
    'svhn': ExperimentConfig(
        name='Split-SVHN',
        dataset='svhn',
        scenario='class_incremental',
        n_tasks=5,
        classes_per_task=2,
        input_dim=32*32*3,
        output_dim=2,
        train=TrainConfig(
            epochs_per_task=5,
            batch_size=128,
            replay_size=20000,
            replay_ratio=1.0,
            kd_weight=2.0,
            kd_temperature=2.0,
            split_thresh=0.01,
            prune_thresh=0.01,
            max_spawn_per_call=5,
        ),
        model=ModelConfig(
            d_latent=64, k_init=32, max_k=256, top_k=8, lora_rank=4
        ),
    ),
}


def run_all_quick(
    benchmarks: List[str] = None,
    backbone: str = 'resnet18',
    seed: int = 42,
    output_dir: str = './quick_results'
) -> Dict:
    """Run quick evaluation on standard benchmarks."""
    benchmarks = benchmarks or ['cifar100', 'cifar10', 'svhn']
    results = {}
    
    for bench_name in benchmarks:
        if bench_name not in QUICK_BENCHMARKS:
            print(f"Unknown benchmark: {bench_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running {bench_name} with {backbone}")
        print(f"{'='*60}")
        
        config = QUICK_BENCHMARKS[bench_name]
        try:
            result = run_quick_experiment(config, backbone, seed, output_dir, verbose=True)
            results[bench_name] = result
        except Exception as e:
            print(f"Error on {bench_name}: {e}")
            results[bench_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("QUICK BENCHMARK SUMMARY")
    print("="*60)
    for bench_name, result in results.items():
        if 'error' not in result:
            m = result['metrics']
            print(f"{bench_name}: Acc={m['avg_final_accuracy']:.2%}, Forgetting={m['avg_forgetting']:.2%}, "
                  f"Head Params={result.get('head_params', 0):,}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmarks', nargs='+', default=['cifar100', 'cifar10', 'svhn'])
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'resnet34', 'vit_b_16', 'mobilenet_v3_small'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='./quick_results')
    args = parser.parse_args()
    
    run_all_quick(args.benchmarks, args.backbone, args.seed, args.output_dir)