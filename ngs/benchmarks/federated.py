"""Federated learning benchmark for NGS with hypernetwork code sharing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
import copy


class FederatedClient:
    """Simulated federated learning client."""
    
    def __init__(self, model, data_loader, lr=0.01, device='cpu'):
        self.model = model
        self.data_loader = data_loader
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    def local_train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                logits = output.logits if hasattr(output, 'logits') else output
                loss = F.cross_entropy(logits, y)
                loss.backward()
                self.optimizer.step()
        return self.get_model_update()
    
    def get_model_update(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def load_model_update(self, state_dict):
        self.model.load_state_dict(state_dict)


def federated_averaging(updates: List[dict]) -> dict:
    """Simple FedAvg aggregation."""
    avg_update = {}
    for key in updates[0]:
        tensors = [u[key] for u in updates]
        if tensors[0].dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, 
                                 torch.complex64, torch.complex128):
            avg_update[key] = torch.stack(tensors).mean(dim=0)
        else:
            # For non-float types (bool, int), use the first client's value
            avg_update[key] = tensors[0].clone()
    return avg_update


def partition_data(x, y, n_clients, data_per_client=200):
    """Partition data among clients."""
    n = len(x)
    indices = torch.randperm(n)
    clients_x, clients_y = [], []
    for i in range(n_clients):
        start = i * data_per_client
        end = min(start + data_per_client, n)
        idx = indices[start:end]
        clients_x.append(x[idx])
        clients_y.append(y[idx])
    return clients_x, clients_y


def run_federated_benchmark(
    n_clients: int = 10,
    n_rounds: int = 50,
    local_epochs: int = 2,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./federated_results",
    latent_dim: int = 32,
) -> Dict[str, Any]:
    """Run federated learning with NGS and code sharing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running federated learning with {n_clients} clients using {device}")

    from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
    from ngs.models import build_ngs

    # Create synthetic data
    n_samples = n_clients * 200
    x_all = torch.randn(n_samples, 28 * 28)
    y_all = torch.randint(0, 10, (n_samples,))

    config = NGSConfig(
        latent_dim=latent_dim,
        k_init=16,
        max_k=128,
        top_k=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
        parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
        topology_control=TopologyControl.DISCRETE_HEURISTIC,
        hypernetwork_code_dim=8,
    )

    global_model = build_ngs(784, 10, config).to(device)
    clients_x, clients_y = partition_data(x_all, y_all, n_clients)

    clients = []
    for i in range(n_clients):
        client_model = copy.deepcopy(global_model)
        client_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(clients_x[i], clients_y[i]),
            batch_size=32, shuffle=True
        )
        clients.append(FederatedClient(client_model, client_loader, device=device))

    round_acc = []
    comm_cost = 0

    for rnd in range(n_rounds):
        # Select subset of clients
        n_selected = max(n_clients // 2, 2)
        selected = np.random.choice(n_clients, n_selected, replace=False)

        updates = []
        for idx in selected:
            update = clients[idx].local_train(local_epochs)
            updates.append(update)
            comm_cost += sum(v.numel() for v in update.values())

        avg_update = federated_averaging(updates)
        global_model.load_state_dict(avg_update)
        for idx in selected:
            clients[idx].load_model_update(avg_update)

        # Evaluate
        global_model.eval()
        test_x = torch.randn(200, 28 * 28).to(device)
        test_y = torch.randint(0, 10, (200,)).to(device)
        with torch.no_grad():
            output = global_model(test_x)
            logits = output.logits if hasattr(output, 'logits') else output
            acc = (logits.argmax(1) == test_y).float().mean().item()
        round_acc.append(acc)

        if rnd % 10 == 0:
            print(f"Round {rnd}: Acc={acc:.4f}, K={global_model.K}, Comm={comm_cost / 1e6:.2f}MB")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "final_acc": round_acc[-1],
        "acc_history": round_acc,
        "total_comm_mb": comm_cost / 1e6,
        "final_k": global_model.K,
    }
    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Final accuracy: {round_acc[-1]:.4f}, Communication: {comm_cost / 1e6:.2f}MB")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_federated_benchmark(args.n_clients, args.n_rounds, device=args.device, seed=args.seed)
