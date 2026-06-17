"""Peer-to-peer Gaussian gossip benchmark for NGS."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from copy import deepcopy


class GaussianGossipNode:
    """A peer node with an NGS model and local data."""

    def __init__(self, node_id: int, model: nn.Module, config, device: str):
        self.node_id = node_id
        self.model = deepcopy(model).to(device)
        self.config = config
        self.device = device
        self.parameters_sent = 0
        self.parameters_received = 0
        self.comm_cost = 0
        self.accuracies = []

    def train_local(self, train_loader, lr: float = 1e-3, epochs: int = 5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.view(x.size(0), -1)
                optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def evaluate(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.view(x.size(0), -1)
                out = self.model(x)
                _, pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        return correct / total

    def extract_codes(self):
        if hasattr(self.model, "param_store") and hasattr(self.model.param_store, "codes"):
            return self.model.param_store.codes.detach().clone()
        return None

    def inject_codes(self, codes):
        if hasattr(self.model, "param_store") and hasattr(self.model.param_store, "codes"):
            self.model.param_store.codes.data.copy_(codes)
            self.parameters_received += codes.numel()
            self.comm_cost += codes.numel() * 4  # 4 bytes per float

    def share_codes(self):
        codes = self.extract_codes()
        if codes is not None:
            self.parameters_sent += codes.numel()
            self.comm_cost += codes.numel() * 4
        return codes


def run_gossip_benchmark(
    n_nodes: int = 5,
    dataset: str = "mnist",
    n_rounds: int = 20,
    local_epochs: int = 3,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./gossip_results",
    latent_dim: int = 32,
    k_init: int = 32,
    max_k: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
    gossip_topology: str = "ring",
    gossip_fraction: float = 0.5,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running gossip benchmark: {n_nodes} nodes, {n_rounds} rounds on {device}")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset, random_split

    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        full_test = datasets.MNIST("./data", train=False, download=True, transform=transform)
        d_in = 28 * 28
        d_out = 10
    elif dataset == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        full_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        full_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        d_in = 28 * 28
        d_out = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Split data among nodes (non-IID)
    n_samples_per_node = len(full_train) // n_nodes
    node_datasets = []
    for i in range(n_nodes):
        start = i * n_samples_per_node
        end = start + n_samples_per_node if i < n_nodes - 1 else len(full_train)
        node_datasets.append(Subset(full_train, list(range(start, end))))

    test_loader = DataLoader(full_test, batch_size=batch_size)

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=k_init,
        max_k=max_k,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        memory_management=MemoryManagement.DYNAMIC_GROWTH,
        split_threshold=0.05,
        prune_threshold=0.01,
        hypernetwork_code_dim=8,
        hypernetwork_hidden_dim=16,
        tau=1.0,
    )

    base_model = build_ngs(d_in, d_out, config)
    nodes = [GaussianGossipNode(i, base_model, config, device) for i in range(n_nodes)]

    if gossip_topology == "ring":
        neighbors = {i: [(i + 1) % n_nodes, (i - 1) % n_nodes] for i in range(n_nodes)}
    elif gossip_topology == "fully_connected":
        neighbors = {i: [j for j in range(n_nodes) if j != i] for i in range(n_nodes)}
    else:
        neighbors = {i: np.random.choice([j for j in range(n_nodes) if j != i], size=min(2, n_nodes - 1), replace=False).tolist() for i in range(n_nodes)}

    acc_history = [[] for _ in range(n_nodes)]
    total_comm_cost = 0

    for round_idx in range(n_rounds):
        for i, node in enumerate(nodes):
            loader = DataLoader(node_datasets[i], batch_size=batch_size, shuffle=True)
            node.train_local(loader, lr=lr, epochs=local_epochs)
            acc = node.evaluate(test_loader)
            acc_history[i].append(acc)

        # Gossip: exchange codes
        for i, node in enumerate(nodes):
            codes_i = node.extract_codes()
            if codes_i is None:
                continue

            n_neighbors = len(neighbors[i])
            if n_neighbors == 0:
                continue

            for j in neighbors[i]:
                codes_j = nodes[j].extract_codes()
                if codes_j is None:
                    continue

                n_params = codes_i.numel()
                avg_codes = (codes_i * (1 - gossip_fraction) + codes_j * gossip_fraction).to(device)
                node.inject_codes(avg_codes)

        round_comm = sum(node.comm_cost for node in nodes) - total_comm_cost
        total_comm_cost = sum(node.comm_cost for node in nodes)
        avg_acc = np.mean([acc_history[i][-1] for i in range(n_nodes)])
        print(f"Round {round_idx}: avg_acc={avg_acc:.4f}, comm_cost={round_comm:.0f} bytes")

    # Centralized baseline
    central_model = build_ngs(d_in, d_out, config).to(device)
    central_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    central_optimizer = torch.optim.AdamW(central_model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(n_rounds * local_epochs):
        for x, y in central_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            central_optimizer.zero_grad()
            out = central_model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            central_optimizer.step()

    central_model.eval()
    central_acc = 0.0
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            out = central_model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        central_acc = correct / total

    results = {
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "topology": gossip_topology,
        "final_avg_acc": float(np.mean([acc_history[i][-1] for i in range(n_nodes)])),
        "centralized_acc": float(central_acc),
        "total_comm_cost": float(total_comm_cost),
        "node_accuracies": [[float(a) for a in acc_history[i]] for i in range(n_nodes)],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"gossip_{n_nodes}nodes_{n_rounds}rounds.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path / f'gossip_{n_nodes}nodes_{n_rounds}rounds.json'}")
    return results
