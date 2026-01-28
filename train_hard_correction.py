#!/usr/bin/env python3
"""
Training Script for Hard Correction Benchmarks
"""

import os
import math
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
from train_continuous import ContinuousModelWrapper


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, default='edelta',
                        choices=['gpt2', 'ddl', 'mhc', 'edelta'])
    parser.add_argument('--difficulty', type=str, default='multiple',
                        choices=['single', 'multiple', 'long_range', 'selective', 'orthogonal'])
    
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
    parser.add_argument('--out_dir', type=str, default='out-hard-correction')
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--eval_iters', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=50)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # E∆-MHC-Geo specific
    parser.add_argument('--gate_reg_weight', type=float, default=0.1)
    parser.add_argument('--init_gate_bias', type=float, default=0.0)
    
    return parser.parse_args()


def load_data(difficulty):
    """Load hard correction dataset."""
    filepath = f'data/correction_hard_{difficulty}.npz'
    data = np.load(filepath)
    
    return {
        'train_x': torch.from_numpy(data['train_x']),
        'train_y': torch.from_numpy(data['train_y']),
        'val_x': torch.from_numpy(data['val_x']),
        'val_y': torch.from_numpy(data['val_y']),
    }


def get_batch(data, split, batch_size, device):
    """Get a batch of data."""
    x = data[f'{split}_x']
    y = data[f'{split}_y']
    
    ix = torch.randint(len(x), (batch_size,))
    x_batch = x[ix].to(device)
    y_batch = y[ix].to(device)
    
    return x_batch, y_batch


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
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
    """Estimate loss on train and val sets."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, split, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def create_model(args, input_dim, block_size):
    """Create model based on model_type."""
    
    if args.model_type == 'gpt2':
        config = BaselineConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout, bias=False, block_size=block_size, vocab_size=1
        )
        core = BaselineGPT(config)
        
    elif args.model_type == 'ddl':
        config = DDLConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout, bias=False, block_size=block_size, vocab_size=1
        )
        core = DDLGPT(config)
        
    elif args.model_type == 'mhc':
        config = mHCConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=args.n_streams, dropout=args.dropout, bias=False,
            block_size=block_size, vocab_size=1
        )
        core = mHCGPT(config)
        
    elif args.model_type == 'edelta':
        config = EdeltaConfig(
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_streams=args.n_streams, dropout=args.dropout, bias=False,
            block_size=block_size, vocab_size=1,
            gate_reg_weight=args.gate_reg_weight, init_gate_bias=args.init_gate_bias
        )
        core = EdeltaGPT(config)
    
    model = ContinuousModelWrapper(core, config, input_dim, block_size)
    return model


def train(args):
    """Main training loop."""
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    print(f"\n=== Loading correction_hard_{args.difficulty} ===")
    data = load_data(args.difficulty)
    
    input_dim = data['train_x'].shape[-1]
    seq_len = data['train_x'].shape[1]
    
    print(f"Input dimension: {input_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Train samples: {len(data['train_x'])}")
    print(f"Val samples: {len(data['val_x'])}")
    
    # Create model
    model = create_model(args, input_dim, seq_len + 1)
    model = model.to(device)
    
    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.95))
    
    # Training
    print(f"\n=== Training ===")
    
    warmup_iters = 100
    best_val_loss = float('inf')
    
    train_log = {'iter': [], 'train_loss': [], 'val_loss': [], 'grad_norm': []}
    
    for iter_num in range(args.max_iters):
        lr = get_lr(iter_num, warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data, args.batch_size, args.eval_iters, device)
            print(f"step {iter_num}: train {losses['train']:.6e}, val {losses['val']:.6e}")
            
            train_log['iter'].append(iter_num)
            train_log['train_loss'].append(losses['train'])
            train_log['val_loss'].append(losses['val'])
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'config': vars(args),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
        
        # Training step
        x, y = get_batch(data, 'train', args.batch_size, device)
        _, loss = model(x, y)
        
        # Add gate regularization for edelta
        if args.model_type == 'edelta' and hasattr(model.core, 'get_gate_regularization_loss'):
            gate_loss = model.core.get_gate_regularization_loss()
            loss = loss + gate_loss
        
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if iter_num % args.log_interval == 0:
            train_log['grad_norm'].append(float(grad_norm))
            print(f"iter {iter_num}: loss {loss.item():.6e}, grad {grad_norm:.4f}")
    
    # Final evaluation
    final_losses = estimate_loss(model, data, args.batch_size, 100, device)
    print(f"\n=== Final ===")
    print(f"Best val loss: {best_val_loss:.6e}")
    print(f"Final val loss: {final_losses['val']:.6e}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'difficulty': args.difficulty,
        'best_val_loss': best_val_loss,
        'final_val_loss': final_losses['val'],
        'final_train_loss': final_losses['train'],
    }
    np.save(os.path.join(args.out_dir, 'results.npy'), results)
    np.save(os.path.join(args.out_dir, 'train_log.npy'), train_log)
    
    return results


if __name__ == '__main__':
    args = get_args()
    train(args)
