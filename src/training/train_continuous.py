"""
Training Script for Continuous Physics Benchmarks

This script trains and evaluates different transformer architectures on the
"Kill-Shot" benchmark datasets designed to expose specific failure modes.

Usage (run from project root with uv):
    # Train baseline GPT on Gyroscope
    uv run src/training/train_continuous.py --model_type gpt2 --dataset gyroscope --out_dir out-baseline
    
    # Train DDL on Gyroscope  
    uv run src/training/train_continuous.py --model_type ddl --dataset gyroscope --out_dir out-ddl
    
    # Train mHC on Gyroscope
    uv run src/training/train_continuous.py --model_type mhc --dataset gyroscope --out_dir out-mhc
    
    # Train E∆-MHC-Geo on Gyroscope
    uv run src/training/train_continuous.py --model_type edelta --dataset gyroscope --out_dir out-proposed

Datasets:
    - gyroscope: Continuous rotation prediction (tests manifold precision)
    - correction: Belief flip / negation (tests topological completeness)
    - stability: Long-horizon identity (tests unconditional isometry)

Models (from src/models/):
    - gpt2: baseline_gpt.py (Standard GPT baseline)
    - ddl: ddl.py (Deep Delta Learning)
    - mhc: mhc.py (DeepSeek mHC with Sinkhorn)
    - edelta: edelta_hybrid.py (E∆-MHC-Geo Hybrid)
"""

import os
import math
import time
import argparse
from contextlib import nullcontext
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing models from src/models
from src.models.baseline_gpt import GPT as BaselineGPT, GPTConfig as BaselineConfig
from src.models.ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from src.models.mhc import GPT as mHCGPT, GPTConfig as mHCConfig
from src.models.edelta_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig


class ContinuousModelWrapper(nn.Module):
    """
    Wrapper that adapts token-based GPT models for continuous vector regression.
    
    Replaces:
        - nn.Embedding → Linear projection (input_dim → model_dim)
        - Softmax head → Linear projection (model_dim → input_dim)
        - CrossEntropy loss → MSE loss
    """
    
    def __init__(self, core_model, config, input_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = config.n_embd
        
        # Input adapter: Raw Vector → Model Dimension
        self.input_proj = nn.Linear(input_dim, config.n_embd)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, config.n_embd))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # The core transformer (bypass embedding, use transformer blocks directly)
        self.core = core_model
        
        # Output adapter: Model Dimension → Raw Vector
        self.output_proj = nn.Linear(config.n_embd, input_dim)
        
        # Initialize projections
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        self.config = config
    
    def forward(self, x, targets=None):
        """
        Forward pass for continuous regression.
        
        Args:
            x: (B, T, input_dim) input vectors
            targets: (B, T, input_dim) target vectors (optional)
            
        Returns:
            logits: (B, T, input_dim) predicted vectors
            loss: MSE loss (if targets provided)
        """
        B, T, _ = x.shape
        
        # Project up to model dimension
        x = self.input_proj(x)  # (B, T, model_dim)
        
        # Add positional encoding
        x = x + self.pos_emb[:, :T, :]
        x = self.dropout(x)
        
        # Pass through transformer blocks (bypass embedding layer)
        for block in self.core.transformer.h:
            x = block(x)
        x = self.core.transformer.ln_f(x)
        
        # Project down to input dimension
        logits = self.output_proj(x)  # (B, T, input_dim)
        
        # Calculate MSE loss
        loss = None
        if targets is not None:
            loss = F.mse_loss(logits, targets)
        
        return logits, loss
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_args():
    parser = argparse.ArgumentParser(description='Train on continuous physics benchmarks')
    
    # Model
    parser.add_argument('--model_type', type=str, default='gpt2',
                        choices=['gpt2', 'ddl', 'mhc', 'edelta'],
                        help='Model architecture to train')
    
    # Data
    parser.add_argument('--dataset', type=str, default='gyroscope',
                        choices=['gyroscope', 'correction', 'stability',
                                 'correction_insight', 'correction_entropy', 'correction_shift',
                                 'near_pi_rotation', 'near_pi_rotation_multiplane'],
                        help='Dataset to train on')
    parser.add_argument('--data_dir', type=str, default='data')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Model size
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Logging
    parser.add_argument('--out_dir', type=str, default='out-continuous')
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--always_save_checkpoint', action='store_true')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    # E∆-MHC-Geo specific (for ablation studies)
    parser.add_argument('--gate_reg_weight', type=float, default=0.1,
                        help='Weight for midpoint collapse regularization (0=disabled)')
    parser.add_argument('--init_gate_bias', type=float, default=0.0,
                        help='Initial gate bias (>0=prefer rotation, <0=prefer reflection)')
    
    # Fair comparison mode
    parser.add_argument('--match_proposed_params', action='store_true',
                        help='Scale up baseline n_layer to match E∆-MHC-Geo params (~1.79M)')
    
    return parser.parse_args()


def create_model(args, input_dim: int, block_size: int):
    """Create wrapped model based on model_type using existing implementations.
    
    When --fair_params is enabled, adjusts n_embd for E∆-MHC-Geo to ensure
    similar parameter counts across models (E∆ uses n_embd=104 when baseline uses 128).
    """
    
    # Fair parameter comparison mode:
    # 
    # --match_proposed_params (RECOMMENDED)
    #   Keeps n_embd=128 for ALL models (same representation dimension)
    #   Increases n_layer for baselines to match E∆-MHC-Geo's ~1.79M params
    #   This tests: does geometric inductive bias beat additional depth?
    #
    # Configuration (all with n_embd=128):
    #   E∆-MHC-Geo: n_layer=6 → 1.788M (reference)
    #   GPT:        n_layer=9 → 1.780M (0.996x)
    #   DDL:        n_layer=8 → 1.784M (0.998x)
    #   mHC:        n_layer=9 → 1.838M (1.028x)
    
    n_embd = args.n_embd
    n_layer = args.n_layer
    use_mhc_projections = True
    geo_hidden_ratio = 4  # Default for E∆-MHC-Geo
    
    # Map baseline n_layer to match E∆-MHC-Geo's param count (keeping n_embd=128)
    BASELINE_NLAYER_FOR_MATCH = {
        'gpt2': 9,   # 1.780M params (0.996x)
        'ddl': 8,    # 1.784M params (0.998x)
        'mhc': 9,    # 1.838M params (1.028x)
    }
    
    if args.match_proposed_params and args.model_type != 'edelta':
        n_layer = BASELINE_NLAYER_FOR_MATCH.get(args.model_type, args.n_layer)
        print(f"\n[MATCH PROPOSED] Scaling up {args.model_type} n_layer: {args.n_layer} → {n_layer}")
        print(f"                 (keeping n_embd={n_embd} for same representation dimension)")
    
    if args.model_type == 'gpt2':
        print(f"\n=== Standard GPT (Baseline) ===")
        config = BaselineConfig(
            n_layer=n_layer,
            n_head=args.n_head,
            n_embd=n_embd,
            dropout=args.dropout,
            bias=False,
            block_size=block_size,
            vocab_size=1,  # Not used in continuous mode
        )
        core = BaselineGPT(config)
        
    elif args.model_type == 'ddl':
        print(f"\n=== Deep Delta Learning (DDL) ===")
        print(f"  Reference: arXiv:2601.00417")
        config = DDLConfig(
            n_layer=n_layer,
            n_head=args.n_head,
            n_embd=n_embd,
            dropout=args.dropout,
            bias=False,
            block_size=block_size,
            vocab_size=1,
        )
        core = DDLGPT(config)
        
    elif args.model_type == 'mhc':
        print(f"\n=== DeepSeek mHC (Data-Dependent) ===")
        print(f"  Reference: arXiv:2512.24880")
        config = mHCConfig(
            n_layer=n_layer,
            n_head=args.n_head,
            n_embd=n_embd,
            n_streams=args.n_streams,
            dropout=args.dropout,
            bias=False,
            block_size=block_size,
            vocab_size=1,
            n_sinkhorn_iters=20,
            alpha_init=0.01,
        )
        core = mHCGPT(config)
        
    elif args.model_type == 'edelta':
        print(f"\n=== E∆-MHC-Geo (Proposed) ===")
        print(f"  init_gate_bias: {args.init_gate_bias}")
        print(f"  gate_reg_weight: {args.gate_reg_weight}")
        config = EdeltaConfig(
            n_layer=n_layer,
            n_head=args.n_head,
            n_embd=n_embd,
            n_streams=args.n_streams,
            dropout=args.dropout,
            bias=False,
            block_size=block_size,
            vocab_size=1,
            gate_reg_weight=args.gate_reg_weight,
            init_gate_bias=args.init_gate_bias,
            use_mhc_projections=use_mhc_projections,
            geo_hidden_ratio=geo_hidden_ratio,
        )
        core = EdeltaGPT(config)
        
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    # Wrap with continuous adapter
    model = ContinuousModelWrapper(core, config, input_dim, max_seq_len=block_size)
    
    n_params = model.get_num_params()
    print(f"  Total parameters: {n_params/1e6:.2f}M")
    
    return model


def load_dataset(dataset_name: str, data_dir: str = 'data') -> dict:
    """Load a continuous dataset from .npy files."""
    path = os.path.join(data_dir, dataset_name)
    
    data = {
        'train_x': torch.from_numpy(np.load(os.path.join(path, 'train_x.npy'))),
        'train_y': torch.from_numpy(np.load(os.path.join(path, 'train_y.npy'))),
        'val_x': torch.from_numpy(np.load(os.path.join(path, 'val_x.npy'))),
        'val_y': torch.from_numpy(np.load(os.path.join(path, 'val_y.npy'))),
    }
    
    # Handle 2D datasets (batch, dim) by adding sequence dimension
    # This makes them compatible with the transformer wrapper which expects (batch, seq, dim)
    for key in ['train_x', 'train_y', 'val_x', 'val_y']:
        if data[key].dim() == 2:
            data[key] = data[key].unsqueeze(1)  # (batch, dim) -> (batch, 1, dim)
    
    # Load additional metadata if available (for paper-style analysis)
    for extra in ['train_entropy', 'train_shift_needed', 'train_scenarios',
                  'train_uncertainty', 'train_expected_shift', 'metadata']:
        extra_path = os.path.join(path, f'{extra}.npy')
        if os.path.exists(extra_path):
            data[extra] = np.load(extra_path, allow_pickle=True)
            if extra == 'metadata':
                data[extra] = data[extra].item()
    
    print(f"Loaded {dataset_name} dataset:")
    print(f"  Train: {data['train_x'].shape} -> {data['train_y'].shape}")
    print(f"  Val: {data['val_x'].shape} -> {data['val_y'].shape}")
    if 'metadata' in data:
        print(f"  Metadata available: {list(data['metadata'].keys())}")
    
    return data


def get_input_dim(dataset_name: str) -> int:
    """
    Get input dimension for each dataset.
    
    Specifications (from comparative study table):
        - Gyroscope:    16-dim vectors
        - Correction:   32-dim vectors (original)
        - Stability:    64-dim vectors
        
    New correction datasets (arXiv:2601.00514v1 inspired):
        - correction_insight:  32-dim (scenario-based correction)
        - correction_entropy:  32-dim (entropy-stratified)
        - correction_shift:    32-dim (shift detection)
    """
    dims = {
        'gyroscope': 16,
        'correction': 32,
        'stability': 64,
        # New correction datasets
        'correction_insight': 32,
        'correction_entropy': 32,
        'correction_shift': 32,
        # Near-π rotation datasets (test Cayley limits)
        'near_pi_rotation': 64,           # Single-plane: γ → 1 (Cayley)
        'near_pi_rotation_multiplane': 64, # Multi-plane: γ → 0 (Householder)
    }
    if dataset_name not in dims:
        # Try to infer from data files
        path = f'data/{dataset_name}/train_x.npy'
        if os.path.exists(path):
            data = np.load(path)
            return data.shape[-1]
        raise ValueError(f"Unknown dataset: {dataset_name}. Expected one of {list(dims.keys())}")
    return dims[dataset_name]


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, 
           learning_rate: float, min_lr: float) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, data, batch_size, eval_iters, device):
    """Estimate loss on train and val splits."""
    model.eval()
    out = {}
    
    for split in ['train', 'val']:
        x_data = data[f'{split}_x'].to(device)
        y_data = data[f'{split}_y'].to(device)
        n_samples = x_data.shape[0]
        
        losses = []
        for _ in range(eval_iters):
            ix = torch.randint(n_samples, (batch_size,))
            x, y = x_data[ix], y_data[ix]
            _, loss = model(x, y)
            losses.append(loss.item())
        
        out[split] = np.mean(losses)
    
    model.train()
    return out


def get_batch(data, split, batch_size, device):
    """Get a random batch from the specified split."""
    x_data = data[f'{split}_x'].to(device)
    y_data = data[f'{split}_y'].to(device)
    n_samples = x_data.shape[0]
    
    ix = torch.randint(n_samples, (batch_size,))
    x, y = x_data[ix], y_data[ix]
    return x, y


def train(args):
    """Main training loop."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load dataset
    print(f"\n=== Loading {args.dataset} dataset ===")
    data = load_dataset(args.dataset, args.data_dir)
    input_dim = get_input_dim(args.dataset)
    seq_len = data['train_x'].shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Sequence length: {seq_len}")
    
    # Create model
    model = create_model(args, input_dim, seq_len + 1)
    model = model.to(device)
    
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    
    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device == 'cuda'))
    
    # Training loop
    print(f"\n=== Training ===")
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Learning rate: {args.learning_rate}")
    
    warmup_iters = 100
    lr_decay_iters = args.max_iters
    
    best_val_loss = float('inf')
    iter_num = 0
    
    # Training log with comprehensive metrics for research papers
    # Following mHC paper (arXiv:2512.24880) metrics
    train_log = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'time': [],
        'grad_norm': [],           # Gradient norm (before clipping)
        'grad_norm_clipped': [],   # Gradient norm (after clipping)
        'param_norm': [],          # Parameter norm
    }
    
    t0 = time.time()
    
    while iter_num < args.max_iters:
        # Get learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters,
                    args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data, args.batch_size, args.eval_iters, device)
            print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
            
            train_log['iter'].append(iter_num)
            train_log['train_loss'].append(losses['train'])
            train_log['val_loss'].append(losses['val'])
            train_log['lr'].append(lr)
            train_log['time'].append(time.time() - t0)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if args.always_save_checkpoint or iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
        
        # Training step
        x, y = get_batch(data, 'train', args.batch_size, device)
        
        with ctx:
            _, loss = model(x, y)
            
            # Add gate regularization loss for edelta
            if args.model_type == 'edelta' and hasattr(model.core, 'get_gate_regularization_loss'):
                gate_loss = model.core.get_gate_regularization_loss()
                loss = loss + gate_loss
        
        scaler.scale(loss).backward()
        
        # Compute gradient norm BEFORE clipping (for stability analysis)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))  # Just compute, don't clip
        
        # Apply gradient clipping
        if args.grad_clip > 0:
            grad_norm_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm_clipped = grad_norm
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Compute parameter norm
        param_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
        
        # Logging with gradient norms (following mHC paper style)
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.6f}, grad_norm {grad_norm:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
            
            # Store gradient metrics
            train_log['grad_norm'].append(float(grad_norm))
            train_log['grad_norm_clipped'].append(float(grad_norm_clipped))
            train_log['param_norm'].append(float(param_norm))
        
        iter_num += 1
    
    # Save final results
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save training log
    np.save(os.path.join(args.out_dir, 'train_log.npy'), train_log)
    
    # Final evaluation
    final_losses = estimate_loss(model, data, args.batch_size, 100, device)
    print(f"Final train loss: {final_losses['train']:.6f}")
    print(f"Final val loss: {final_losses['val']:.6f}")
    
    results = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_losses['train'],
        'final_val_loss': final_losses['val'],
        'n_params': model.get_num_params(),
        'config': vars(args),
    }
    np.save(os.path.join(args.out_dir, 'results.npy'), results)
    
    return results


if __name__ == '__main__':
    args = get_args()
    train(args)
