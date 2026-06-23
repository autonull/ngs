"""TinyShakespeare FFN swap benchmark (Experiment 2A).
Tests NGS as Transformer FFN replacement with matched capacity.
Target: Match 10.81 perplexity with fewer params."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
import math


def compute_perplexity(model, loader, device, criterion):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # x: [B, seq_len, vocab_size] (one-hot) -> flatten to [B, seq_len * vocab_size]
            B, seq_len, vocab_size = x.shape
            x_flat = x.view(B, -1)
            logits = model(x_flat)  # [B, vocab_size]
            loss = criterion(logits, y)
            total_loss += loss.item() * B
            total_tokens += B
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


class TransformerFFN(nn.Module):
    """Standard Transformer FFN (Gated Linear Unit style)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NGSFFN(nn.Module):
    """NGS as FFN replacement (position-wise)."""
    def __init__(self, d_model: int, config):
        super().__init__()
        self.d_model = d_model
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        from ngs.models import build_ngs
        
        # NGS with input=output=d_model
        self.ngs = build_ngs(d_model, d_model, config)
    
    def forward(self, x):
        # x: [B, seq_len, d_model] -> apply NGS per position
        B, seq_len, d_model = x.shape
        x_flat = x.view(B * seq_len, d_model)
        out_obj = self.ngs(x_flat)
        logits = out_obj.logits
        return logits.view(B, seq_len, d_model)


class TinyTransformer(nn.Module):
    """Minimal Transformer for character-level LM."""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 d_ff: int, max_seq_len: int, use_ngs_ffn: bool = False, ngs_config=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embedding (from one-hot)
        self.input_proj = nn.Linear(vocab_size, d_model, bias=False)
        
        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            if use_ngs_ffn:
                layer['ffn'] = NGSFFN(d_model, ngs_config)
            else:
                layer['ffn'] = TransformerFFN(d_model, d_ff)
            self.layers.append(layer)
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # x: [B, seq_len * vocab_size] -> [B, seq_len, vocab_size]
        B = x.size(0)
        x = x.view(B, self.max_seq_len, self.vocab_size)
        
        # Project to d_model
        x = self.input_proj(x)  # [B, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pe.unsqueeze(0)
        
        # Transformer layers
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x, _ = layer['attn'](x, x, x)
            x = x + residual
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = x + residual
        
        x = self.output_norm(x)
        # Use last token for next-char prediction
        logits = self.output_proj(x[:, -1])  # [B, vocab_size]
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_tinyshakespeare_ffn_benchmark(
    use_ngs_ffn: bool = False,
    n_tasks: int = 1,
    epochs: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "./tinyshakespeare_ffn_results",
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 1024,
    seq_len: int = 16,
    batch_size: int = 256,
    lr: float = 3e-4,
    **kwargs
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Running TinyShakespeare FFN benchmark: NGS_FFN={use_ngs_ffn}")

    from experiments.datasets_tinyshakespeare import load_tinyshakespeare, create_sequences
    from torch.utils.data import DataLoader, TensorDataset

    # Load data
    encoded, stoi, itos = load_tinyshakespeare('./data', seq_len)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    X, y = create_sequences(encoded, seq_len)
    
    # Use subset for quick experiments (default 50000 train, 5000 test)
    max_train = kwargs.get('max_train', 50000)
    max_test = kwargs.get('max_test', 5000)
    n_train = min(int(0.9 * len(X)), max_train)
    n_test = min(len(X) - n_train, max_test)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:n_train + n_test], y[n_train:n_train + n_test]

    # One-hot encode
    X_train_oh = np.zeros((len(X_train), seq_len, vocab_size), dtype=np.float32)
    X_test_oh = np.zeros((len(X_test), seq_len, vocab_size), dtype=np.float32)
    X_train_oh[np.arange(len(X_train))[:, None], np.arange(seq_len), X_train] = 1.0
    X_test_oh[np.arange(len(X_test))[:, None], np.arange(seq_len), X_test] = 1.0

    train_ds = TensorDataset(torch.from_numpy(X_train_oh).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test_oh).float(), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)

    # NGS config for FFN
    if use_ngs_ffn:
        from ngs.core import NGSConfig, RoutingStrategy, ParameterStorage, TopologyControl, MemoryManagement
        ngs_config = NGSConfig(
            latent_dim=d_model // 4,  # 64
            k_init=32,
            max_k=128,
            top_k=4,
            routing=RoutingStrategy.MONOLITHIC_MAHALANOBIS,
            parameter_storage=ParameterStorage.HYPERNETWORK_GENERATED,
            topology_control=TopologyControl.DISCRETE_HEURISTIC,
            memory_management=MemoryManagement.DYNAMIC,
            hypernetwork_code_dim=16,
            hypernetwork_hidden_dim=64,
            tau=1.0,
        )
    else:
        ngs_config = None

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=seq_len,
        use_ngs_ffn=use_ngs_ffn,
        ngs_config=ngs_config
    ).to(device)

    total_params = count_params(model)
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_ppl = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            x_flat = x.view(B, -1)
            
            optimizer.zero_grad()
            logits = model(x_flat)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_ppl = math.exp(epoch_loss / len(train_loader))
        test_ppl = compute_perplexity(model, test_loader, device, criterion)
        
        print(f"Epoch {epoch}: train_ppl={train_ppl:.2f}, test_ppl={test_ppl:.2f}")

        if test_ppl < best_ppl:
            best_ppl = test_ppl

    results = {
        "use_ngs_ffn": use_ngs_ffn,
        "total_params": total_params,
        "best_test_ppl": float(best_ppl),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = "ngs" if use_ngs_ffn else "dense"
    with open(Path(output_dir) / f"tinyshakespeare_{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ngs-ffn", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    run_tinyshakespeare_ffn_benchmark(
        use_ngs_ffn=args.use_ngs_ffn,
        epochs=args.epochs,
        device=args.device
    )