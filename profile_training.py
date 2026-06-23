"""
Line-by-line training profile to identify bottlenecks.
"""
import sys, os, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ngs.modules.ngs_layer import build_stacked_ngs
from experiments.ngs_layer_runner import _load_standard_dataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def profile_training():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load dataset
    train_ds, test_ds, input_dim, output_dim, _ = _load_standard_dataset('cifar10')
    batch_size = min(256, max(32, len(train_ds) // 50))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Build model
    model = build_stacked_ngs(d_in=input_dim, d_out=output_dim, n_layers=3, d_latent=128, n_experts=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {batch_size}, Batches/epoch: {len(train_loader)}")
    print()
    
    # Profile single epoch
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.time()
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        if i >= 5:  # Profile first 5 batches
            break
        x, y = x.to(device), y.to(device)
        
        t_batch = time.time()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if i == 0:
            print(f"  Batch {i}: {time.time() - t_batch:.3f}s (first batch includes compilation)")
        else:
            print(f"  Batch {i}: {time.time() - t_batch:.3f}s")
    
    profiler.disable()
    epoch_time = time.time() - t0
    print(f"\n5 batches: {epoch_time:.2f}s, avg/batch: {epoch_time/5:.3f}s")
    print(f"Extrapolated full epoch ({len(train_loader)} batches): {epoch_time/5 * len(train_loader):.1f}s")
    print(f"10 epochs: {epoch_time/5 * len(train_loader) * 10 / 60:.1f} min")
    
    # Print top functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print("\n=== TOP 30 FUNCTIONS (cumulative) ===")
    print(s.getvalue())
    
    # Also by self time
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('time')
    ps2.print_stats(30)
    print("\n=== TOP 30 FUNCTIONS (self time) ===")
    print(s2.getvalue())


def profile_forward_backward():
    """Profile just forward + backward for a single batch."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_ds, _, input_dim, output_dim, _ = _load_standard_dataset('cifar10')
    x = train_ds[0][0].unsqueeze(0).to(device)
    y = torch.tensor([train_ds[0][1]]).to(device)
    
    model = build_stacked_ngs(d_in=input_dim, d_out=output_dim, n_layers=3, d_latent=128, n_experts=256).to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(x)
    
    # Profile forward
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        logits = model(x)
    torch.cuda.synchronize()
    forward_time = (time.time() - t0) / 100 * 1000
    print(f"Forward (100 iters): {forward_time:.2f} ms/batch")
    
    # Profile forward + backward
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
    torch.cuda.synchronize()
    fb_time = (time.time() - t0) / 100 * 1000
    print(f"Forward+Backward (100 iters): {fb_time:.2f} ms/batch")
    print(f"Backward only: {fb_time - forward_time:.2f} ms/batch")
    
    # GPU memory
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == '__main__':
    print("=" * 60)
    print("MICRO-BENCHMARK: Forward/Backward")
    print("=" * 60)
    profile_forward_backward()
    
    print("\n" + "=" * 60)
    print("MACRO-BENCHMARK: Training Loop (5 batches)")
    print("=" * 60)
    profile_training()