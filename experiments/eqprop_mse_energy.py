#!/usr/bin/env python
"""
MSE Energy Ablation for EqNGS — TODO11 Phase A1.4 / B1.1

Replace Mahalanobis routing energy with MSE internal energy.
If this fixes the 67% plateau, the energy function is the sole culprit.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/me/ngs')
sys.path.insert(0, '/home/me/ngs/bioplausible/mep')

from ngs.models.ngs import NGSModel
from ngs.core.interfaces import NGSConfig, RoutingStrategy
from ngs.modules.eqprop import EqNGSLayer, create_eqngs


class EqNGSLayerMSE(EqNGSLayer):
    """
    EqNGSLayer variant that uses MSE internal energy instead of Mahalanobis.
    
    Energy = 0.5 * MSE(z, p_up(p_down(x))) + beta * CE(output, target)
    
    This mimics the bioplausible smep energy function.
    """
    
    def _compute_routing_energy(self, z, router_out, target=None, beta=0.0):
        """
        Compute MSE-based internal energy instead of Mahalanobis.
        
        Args:
            z: Latent representation [B, d]
            router_out: Router output (unused in MSE energy)
            target: Target labels [B] or None
            beta: Nudging strength
            
        Returns:
            Scalar energy
        """
        # Reconstruction: p_up(p_down(x)) should approximate x in latent space
        # But we don't have x here, only z. So use autoencoder-style:
        # Project z back through p_up and p_down (but we don't have p_down in EqNGSLayer)
        # Instead, use the router's forward pass as "reconstruction target"
        
        # For MSE energy, we need a "target" state for each layer.
        # The bioplausible approach: state should match h(state_prev)
        # Here: z is the input to router, router produces blended output
        
        # Simple approach: MSE between z and the blended output from router
        # This encourages the router to preserve information
        
        if router_out is None:
            return torch.tensor(0.0, device=z.device)
        
        # Get blended output from router
        if hasattr(router_out, 'indices') and hasattr(router_out, 'weights'):
            # Standard routing
            active_indices = router_out.indices
            weights = router_out.weights
            
            # Get local outputs from param_store
            local_out = self.ngs.param_store(active_indices, z)  # [B, K, d]
            
            # Weighted combination
            blended = (weights.unsqueeze(-1) * local_out).sum(dim=1)  # [B, d]
            
            # MSE internal energy: want z to be close to blended (information preservation)
            # Or want blended to be close to z
            internal_energy = 0.5 * F.mse_loss(blended, z, reduction='mean')
        else:
            internal_energy = torch.tensor(0.0, device=z.device)
        
        # Nudge energy
        nudge_energy = torch.tensor(0.0, device=z.device)
        if target is not None and beta > 0:
            # Use the standard logits computation
            if hasattr(router_out, 'indices') and hasattr(router_out, 'weights'):
                active_indices = router_out.indices
                weights = router_out.weights
                local_out = self.ngs.param_store(active_indices, z)
                blended = (weights.unsqueeze(-1) * local_out).sum(dim=1)
                out = self.ngs.p_up(blended + self.ngs.gamma * z)
                nudge_energy = beta * F.cross_entropy(out, target)
        
        return internal_energy + nudge_energy


def create_eqngs_mse(
    d_in: int,
    d_out: int,
    config: NGSConfig,
    ep_beta: float = 0.5,
    ep_settle_steps: int = 10,
    ep_settle_lr: float = 0.2,
    spectral_mode: str = 'none',
) -> nn.Module:
    """Create EqNGS with MSE internal energy."""
    # We'll monkey-patch the energy function
    eqngs = create_eqngs(
        d_in, d_out, config,
        ep_beta=ep_beta,
        ep_settle_steps=ep_settle_steps,
        ep_settle_lr=ep_settle_lr,
        spectral_mode=spectral_mode,
    )
    
    # Replace the energy function
    def mse_energy(z, router_out, target=None, beta=0.0):
        if router_out is None:
            return torch.tensor(0.0, device=z.device)
        
        if hasattr(router_out, 'indices') and hasattr(router_out, 'weights'):
            active_indices = router_out.indices
            weights = router_out.weights
            local_out = eqngs.ngs.param_store(active_indices, z)
            blended = (weights.unsqueeze(-1) * local_out).sum(dim=1)
            internal_energy = 0.5 * F.mse_loss(blended, z, reduction='mean')
        else:
            internal_energy = torch.tensor(0.0, device=z.device)
        
        nudge_energy = torch.tensor(0.0, device=z.device)
        if target is not None and beta > 0:
            if hasattr(router_out, 'indices') and hasattr(router_out, 'weights'):
                active_indices = router_out.indices
                weights = router_out.weights
                local_out = eqngs.ngs.param_store(active_indices, z)
                blended = (weights.unsqueeze(-1) * local_out).sum(dim=1)
                out = eqngs.ngs.p_up(blended + eqngs.ngs.gamma * z)
                nudge_energy = beta * F.cross_entropy(out, target)
        
        return internal_energy + nudge_energy
    
    eqngs._compute_routing_energy = mse_energy
    return eqngs


def get_mnist_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train = datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
    test = datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser(description="EqNGS with MSE energy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--ep-beta", type=float, default=0.5)
    parser.add_argument("--settle-lr", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", default="results/diagnostics/eqprop_mse_energy.json")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Settle steps: {args.settle_steps}")
    print(f"EP beta: {args.ep_beta}")
    print(f"Settle LR: {args.settle_lr}")
    
    config = NGSConfig(
        latent_dim=64,
        max_k=32,
        top_k=8,
        k_init=8,
        routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
    )
    
    # Create EqNGS with MSE energy
    eqngs = create_eqngs_mse(
        d_in=784,
        d_out=10,
        config=config,
        ep_beta=args.ep_beta,
        ep_settle_steps=args.settle_steps,
        ep_settle_lr=args.settle_lr,
        spectral_mode='none',
    ).to(device)
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    
    results = {
        "config": {
            "mode": "ep_mse_energy",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "beta": args.ep_beta,
            "settle_steps": args.settle_steps,
            "settle_lr": args.settle_lr,
            "latent_dim": config.latent_dim,
            "max_k": config.max_k,
            "top_k": config.top_k,
        },
        "epochs": [],
        "final_accuracy": 0.0,
    }
    
    for epoch in range(args.epochs):
        eqngs.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            result = eqngs.ep_step(x, y)
            
            epoch_loss += result.get('loss', 0.0)
            epoch_acc += result.get('accuracy', 0.0)
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        avg_acc = epoch_acc / n_batches if n_batches > 0 else 0.0
        
        # Test accuracy
        eqngs.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = eqngs(x)
                logits = out.logits if hasattr(out, 'logits') else out
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, test_acc={test_acc:.4f}")
        
        results["epochs"].append({
            "epoch": epoch,
            "loss": avg_loss,
            "train_accuracy": avg_acc,
            "test_accuracy": test_acc,
        })
        
        results["final_accuracy"] = test_acc
    
    print(f"\nFinal test accuracy: {results['final_accuracy']:.4f}")
    
    # Gate check
    if results["final_accuracy"] >= 0.85:
        print("\n✓ MSE energy achieves >85% - energy function is the root cause!")
    elif results["final_accuracy"] >= 0.70:
        print("\n⚠ MSE energy >70% but <85% - partial fix")
    else:
        print("\n✗ MSE energy <70% - energy function NOT the sole culprit")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()