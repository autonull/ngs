"""Unified experiment configuration loader from YAML."""
from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import torch

from ngs.core.interfaces import (
    NGSConfig, RoutingStrategy, ParameterStorage, 
    TopologyControl, MemoryManagement
)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    dataset: str
    scenario: str
    n_tasks: int
    classes_per_task: int
    input_dim: int
    output_dim: int
    model: NGSConfig = field(default_factory=lambda: NGSConfig())
    training: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    
    # Parse model config
    model_data = data.get('model', {})
    routing_str = model_data.get('routing', 'monolithic_mahalanobis')
    param_storage_str = model_data.get('parameter_storage', 'direct_adapter')
    topology_str = model_data.get('topology_control', 'discrete_heuristic')
    memory_str = model_data.get('memory_management', 'pre_allocated')
    
    model_config = NGSConfig(
        latent_dim=model_data.get('latent_dim', 32),
        k_init=model_data.get('k_init', 128),
        max_k=model_data.get('max_k', 512),
        top_k=model_data.get('top_k', 8),
        routing=RoutingStrategy(routing_str),
        parameter_storage=ParameterStorage(param_storage_str),
        topology_control=TopologyControl(topology_str),
        memory_management=MemoryManagement(memory_str),
        num_subspaces=model_data.get('num_subspaces', 4),
        hypernetwork_code_dim=model_data.get('hypernetwork_code_dim', 8),
        split_threshold=model_data.get('split_threshold', 0.05),
        prune_threshold=model_data.get('prune_threshold', 0.01),
    )
    
    return ExperimentConfig(
        name=data.get('experiment', path.stem),
        dataset=data.get('dataset', ''),
        scenario=data.get('scenario', 'class_incremental'),
        n_tasks=data.get('n_tasks', 5),
        classes_per_task=data.get('classes_per_task', 2),
        input_dim=data.get('input_dim', 784),
        output_dim=data.get('output_dim', 10),
        model=model_config,
        training=data.get('training', {}),
        seeds=data.get('seeds', [42, 123, 456]),
        device=data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
    )


def discover_configs(config_dir: str | Path = "configs") -> Dict[str, Path]:
    """Discover all YAML config files organized by category."""
    config_dir = Path(config_dir)
    configs = {}
    for yaml_file in config_dir.rglob("*.yaml"):
        rel = yaml_file.relative_to(config_dir)
        configs[str(rel)] = yaml_file
    return configs


def load_all_configs(config_dir: str | Path = "configs") -> Dict[str, ExperimentConfig]:
    """Load all experiment configs from directory."""
    configs = discover_configs(config_dir)
    return {name: load_experiment_config(path) for name, path in configs.items()}


def config_to_train_kwargs(cfg: ExperimentConfig) -> dict:
    """Convert ExperimentConfig training section to trainer kwargs."""
    t = cfg.training
    return {
        'lr': t.get('lr', 1e-3),
        'weight_decay': t.get('weight_decay', 1e-4),
        'epochs': t.get('epochs_per_task', 5),
        'batch_size': t.get('batch_size', 256),
        'replay_size': t.get('replay_size', 50000),
        'replay_ratio': t.get('replay_ratio', 1.0),
        'kd_weight': t.get('kd_weight', 10.0),
        'kd_temperature': t.get('kd_temperature', 2.0),
        'entropy_weight': t.get('entropy_weight', 0.01),
        'diversity_weight': t.get('diversity_weight', 0.01),
        'adapt_every_epoch': t.get('adapt_every_epoch', True),
        'split_thresh': cfg.model.split_threshold,
        'prune_thresh': cfg.model.prune_threshold,
    }
