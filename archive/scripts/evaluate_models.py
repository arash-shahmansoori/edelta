"""
Evaluation Script for Physics Benchmark Experiments

This script provides detailed analysis and visualization of model performance
on the three "Kill-Shot" benchmark datasets.

Usage:
    # Evaluate all trained models
    python evaluate_models.py --dataset gyroscope
    
    # Run stability test (long-horizon inference)
    python evaluate_models.py --dataset stability --long_horizon
    
    # Generate comparison plots
    python evaluate_models.py --dataset gyroscope --plot

Outputs:
    - Chart 1: Gyroscope - MSE vs Rotation Angle
    - Chart 2: Correction - Cosine Similarity vs Time Step
    - Chart 3: Stability - Norm vs Number of Recursive Passes
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import existing models
from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig


class ContinuousModelWrapper(nn.Module):
    """Wrapper for continuous regression (same as in train_continuous.py)."""
    
    def __init__(self, core_model, config, input_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = config.n_embd
        self.input_proj = nn.Linear(input_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, config.n_embd))
        self.dropout = nn.Dropout(config.dropout)
        self.core = core_model
        self.output_proj = nn.Linear(config.n_embd, input_dim)
        self.config = config
    
    def forward(self, x, targets=None):
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]
        x = self.dropout(x)
        for block in self.core.transformer.h:
            x = block(x)
        x = self.core.transformer.ln_f(x)
        logits = self.output_proj(x)
        loss = F.mse_loss(logits, targets) if targets is not None else None
        return logits, loss


def load_dataset(dataset_name: str, data_dir: str = 'data') -> dict:
    """Load dataset from .npy files."""
    path = os.path.join(data_dir, dataset_name)
    return {
        'train_x': torch.from_numpy(np.load(os.path.join(path, 'train_x.npy'))),
        'train_y': torch.from_numpy(np.load(os.path.join(path, 'train_y.npy'))),
        'val_x': torch.from_numpy(np.load(os.path.join(path, 'val_x.npy'))),
        'val_y': torch.from_numpy(np.load(os.path.join(path, 'val_y.npy'))),
    }


def get_input_dim(dataset_name: str) -> int:
    """
    Get input dimension for each dataset.
    
    Specifications (from comparative study table):
        - Gyroscope:  16-dim vectors
        - Correction: 32-dim vectors
        - Stability:  64-dim vectors
    """
    dims = {'gyroscope': 16, 'correction': 32, 'stability': 64}
    if dataset_name not in dims:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dims[dataset_name]


def load_model(model_type: str, checkpoint_path: str, input_dim: int, 
               seq_len: int, device: str = 'cuda'):
    """Load a trained model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint['config']
    
    if model_type == 'gpt2':
        config = BaselineConfig(
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embd=cfg['n_embd'],
            dropout=cfg['dropout'], bias=False, block_size=seq_len + 1, vocab_size=1,
        )
        core = BaselineGPT(config)
    elif model_type == 'ddl':
        config = DDLConfig(
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embd=cfg['n_embd'],
            dropout=cfg['dropout'], bias=False, block_size=seq_len + 1, vocab_size=1,
        )
        core = DDLGPT(config)
    elif model_type == 'mhc':
        config = mHCConfig(
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embd=cfg['n_embd'],
            n_streams=cfg['n_streams'], dropout=cfg['dropout'], bias=False,
            block_size=seq_len + 1, vocab_size=1, n_sinkhorn_iters=20, alpha_init=0.01,
        )
        core = mHCGPT(config)
    elif model_type == 'edelta':
        config = EdeltaConfig(
            n_layer=cfg['n_layer'], n_head=cfg['n_head'], n_embd=cfg['n_embd'],
            n_streams=cfg['n_streams'], dropout=cfg['dropout'], bias=False,
            block_size=seq_len + 1, vocab_size=1, gate_reg_weight=cfg.get('gate_reg_weight', 0.01),
        )
        core = EdeltaGPT(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = ContinuousModelWrapper(core, config, input_dim, max_seq_len=seq_len + 1)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_gyroscope(model: nn.Module, data_dir: str = 'data/gyroscope',
                       device: str = 'cuda') -> Dict[str, np.ndarray]:
    """Evaluate model on Gyroscope dataset. Returns MSE breakdown by rotation angle."""
    
    val_x = torch.from_numpy(np.load(os.path.join(data_dir, 'val_x.npy'))).to(device)
    val_y = torch.from_numpy(np.load(os.path.join(data_dir, 'val_y.npy'))).to(device)
    val_theta = np.load(os.path.join(data_dir, 'val_theta.npy'))
    
    mse_per_sample = []
    for i in range(val_x.shape[0]):
        pred, _ = model(val_x[i:i+1], val_y[i:i+1])
        mse = F.mse_loss(pred, val_y[i:i+1]).item()
        mse_per_sample.append(mse)
    
    mse_per_sample = np.array(mse_per_sample)
    
    # Bin by rotation angle
    angle_bins = np.linspace(0, 2.5, 11)
    angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    mse_by_angle = []
    
    for i in range(len(angle_bins) - 1):
        mask = (val_theta >= angle_bins[i]) & (val_theta < angle_bins[i+1])
        mse_by_angle.append(mse_per_sample[mask].mean() if mask.sum() > 0 else np.nan)
    
    return {
        'angle_centers': angle_centers,
        'mse_by_angle': np.array(mse_by_angle),
        'mse_all': mse_per_sample.mean(),
        'thetas': val_theta,
        'mse_per_sample': mse_per_sample,
    }


@torch.no_grad()
def evaluate_correction(model: nn.Module, data_dir: str = 'data/correction',
                        device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Evaluate model on Correction dataset.
    
    Returns cosine similarity over time, with special attention to the flip point.
    The signal position varies per sequence, so we also compute flip accuracy.
    """
    val_x = torch.from_numpy(np.load(os.path.join(data_dir, 'val_x.npy'))).to(device)
    val_y = torch.from_numpy(np.load(os.path.join(data_dir, 'val_y.npy'))).to(device)
    
    # Load signal positions if available
    signal_pos_path = os.path.join(data_dir, 'signal_positions.npy')
    if os.path.exists(signal_pos_path):
        all_signal_positions = np.load(signal_pos_path)
        # Get validation portion (last n_val samples)
        metadata_path = os.path.join(data_dir, 'metadata.npy')
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
            n_train = metadata.get('n_train', 4500)
            signal_positions = all_signal_positions[n_train:]
        else:
            signal_positions = all_signal_positions[-val_x.shape[0]:]
        mean_signal_pos = int(signal_positions.mean())
    else:
        # Fallback: assume signal at middle
        mean_signal_pos = val_x.shape[1] // 2
        signal_positions = None
    
    pred, _ = model(val_x, val_y)
    B, T, D = pred.shape
    
    # Overall cosine similarity per time step
    cosine_sim = F.cosine_similarity(pred, val_y, dim=-1)
    cosine_sim_mean = cosine_sim.mean(dim=0).cpu().numpy()
    cosine_sim_std = cosine_sim.std(dim=0).cpu().numpy()
    mse = F.mse_loss(pred, val_y).item()
    
    # Compute flip accuracy at each sample's signal position
    if signal_positions is not None:
        flip_accuracies = []
        for i, sig_pos in enumerate(signal_positions):
            if sig_pos < T:
                # At signal position, target should be -concept (cosine sim to target = 1.0 if perfect)
                flip_cos = F.cosine_similarity(
                    pred[i, sig_pos:sig_pos+1], 
                    val_y[i, sig_pos:sig_pos+1], 
                    dim=-1
                ).item()
                flip_accuracies.append(flip_cos)
        flip_accuracy_mean = np.mean(flip_accuracies) if flip_accuracies else 0.0
    else:
        flip_accuracy_mean = cosine_sim_mean[mean_signal_pos] if mean_signal_pos < T else 0.0
    
    return {
        'time_steps': np.arange(T),
        'cosine_sim_mean': cosine_sim_mean,
        'cosine_sim_std': cosine_sim_std,
        'mse': mse,
        'signal_position': mean_signal_pos,
        'flip_accuracy': flip_accuracy_mean,
    }


@torch.no_grad()
def evaluate_stability_long_horizon(model: nn.Module, data_dir: str = 'data/stability',
                                     n_steps: int = 10000, device: str = 'cuda'):
    """Evaluate stability over very long horizon (autoregressive)."""
    
    keys = torch.from_numpy(np.load(os.path.join(data_dir, 'val_keys.npy'))).to(device)
    x = keys.unsqueeze(1)  # (N, 1, D)
    
    norms = [torch.norm(x, dim=-1).squeeze().cpu().numpy()]
    
    print(f"Running {n_steps} autoregressive steps...")
    for step in range(n_steps):
        pred, _ = model(x, None)
        x = pred
        norms.append(torch.norm(x, dim=-1).squeeze().cpu().numpy())
        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}/{n_steps}, mean norm: {norms[-1].mean():.4f}")
    
    norms = np.stack(norms)
    return {
        'steps': np.arange(n_steps + 1),
        'norms_mean': norms.mean(axis=1),
        'norms_std': norms.std(axis=1),
        'norms_all': norms,
        'initial_norm': norms[0].mean(),
        'final_norm': norms[-1].mean(),
    }


def plot_gyroscope_comparison(results: Dict[str, Dict], save_path: str = 'gyroscope_comparison.png'):
    """Generate Chart 1: MSE vs Rotation Angle."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'gpt2': 'blue', 'ddl': 'orange', 'mhc': 'green', 'edelta': 'red'}
        labels = {'gpt2': 'Standard GPT', 'ddl': 'DDL', 'mhc': 'DeepSeek mHC', 'edelta': 'E∆-MHC-Geo'}
        
        for model_type, data in results.items():
            ax.plot(data['angle_centers'], data['mse_by_angle'], 
                    marker='o', color=colors.get(model_type, 'gray'),
                    label=f"{labels.get(model_type, model_type)} (avg: {data['mse_all']:.4f})")
        
        ax.set_xlabel('Rotation Angle θ (radians)')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Gyroscope: MSE vs Rotation Angle (Lower is better)')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    except ImportError:
        print("matplotlib not available")


def plot_correction_comparison(results: Dict[str, Dict], save_path: str = 'correction_comparison.png'):
    """Generate Chart 2: Cosine Similarity vs Time Step."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'gpt2': 'blue', 'ddl': 'orange', 'mhc': 'green', 'edelta': 'red'}
        labels = {'gpt2': 'Standard GPT', 'ddl': 'DDL', 'mhc': 'DeepSeek mHC', 'edelta': 'E∆-MHC-Geo'}
        
        for model_type, data in results.items():
            ax.plot(data['time_steps'], data['cosine_sim_mean'],
                    marker='o', color=colors.get(model_type, 'gray'),
                    label=labels.get(model_type, model_type))
            ax.fill_between(data['time_steps'],
                           data['cosine_sim_mean'] - data['cosine_sim_std'],
                           data['cosine_sim_mean'] + data['cosine_sim_std'],
                           alpha=0.2, color=colors.get(model_type, 'gray'))
        
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Signal (Flip)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cosine Similarity to Target')
        ax.set_title('Correction Protocol: Cosine Similarity vs Time (Higher is better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    except ImportError:
        print("matplotlib not available")


def plot_stability_comparison(results: Dict[str, Dict], save_path: str = 'stability_comparison.png'):
    """Generate Chart 3: Norm vs Number of Steps."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'gpt2': 'blue', 'ddl': 'orange', 'mhc': 'green', 'edelta': 'red'}
        labels = {'gpt2': 'Standard GPT', 'ddl': 'DDL', 'mhc': 'DeepSeek mHC', 'edelta': 'E∆-MHC-Geo'}
        
        for model_type, data in results.items():
            ax.plot(data['steps'], data['norms_mean'],
                    color=colors.get(model_type, 'gray'),
                    label=f"{labels.get(model_type, model_type)} (final: {data['final_norm']:.4f})")
            ax.fill_between(data['steps'],
                           data['norms_mean'] - data['norms_std'],
                           data['norms_mean'] + data['norms_std'],
                           alpha=0.2, color=colors.get(model_type, 'gray'))
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Target ||v|| = 1')
        ax.set_xlabel('Number of Autoregressive Steps')
        ax.set_ylabel('Vector Norm ||v||')
        ax.set_title('Stability: Norm Preservation (Horizontal at 1.0 is ideal)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    except ImportError:
        print("matplotlib not available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['gyroscope', 'correction', 'stability'])
    parser.add_argument('--model_dirs', nargs='+', 
                        default=['out-baseline', 'out-ddl', 'out-mhc', 'out-proposed'])
    parser.add_argument('--model_types', nargs='+',
                        default=['gpt2', 'ddl', 'mhc', 'edelta'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--long_horizon', action='store_true')
    parser.add_argument('--n_steps', type=int, default=10000)
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    results = {}
    
    for model_type, model_dir in zip(args.model_types, args.model_dirs):
        ckpt_path = os.path.join(model_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue
        
        print(f"\n=== Evaluating {model_type} ===")
        
        data = load_dataset(args.dataset, args.data_dir)
        input_dim = get_input_dim(args.dataset)
        seq_len = data['train_x'].shape[1]
        
        model = load_model(model_type, ckpt_path, input_dim, seq_len, args.device)
        data_path = os.path.join(args.data_dir, args.dataset)
        
        if args.dataset == 'gyroscope':
            results[model_type] = evaluate_gyroscope(model, data_path, args.device)
            print(f"  Overall MSE: {results[model_type]['mse_all']:.6f}")
        elif args.dataset == 'correction':
            results[model_type] = evaluate_correction(model, data_path, args.device)
            print(f"  MSE: {results[model_type]['mse']:.6f}")
        elif args.dataset == 'stability':
            if args.long_horizon:
                results[model_type] = evaluate_stability_long_horizon(
                    model, data_path, args.n_steps, args.device)
            else:
                val_x = torch.from_numpy(np.load(os.path.join(data_path, 'val_x.npy'))).to(args.device)
                val_y = torch.from_numpy(np.load(os.path.join(data_path, 'val_y.npy'))).to(args.device)
                _, loss = model(val_x, val_y)
                results[model_type] = {'mse': loss.item()}
                print(f"  MSE: {loss.item():.6f}")
    
    np.save(f'{args.dataset}_results.npy', results)
    
    if args.plot:
        if args.dataset == 'gyroscope':
            plot_gyroscope_comparison(results)
        elif args.dataset == 'correction':
            plot_correction_comparison(results)
        elif args.dataset == 'stability' and args.long_horizon:
            plot_stability_comparison(results)
    
    print("\n=== Summary ===")
    for model_type, data in results.items():
        if 'mse_all' in data:
            print(f"{model_type}: MSE = {data['mse_all']:.6f}")
        elif 'mse' in data:
            print(f"{model_type}: MSE = {data['mse']:.6f}")


if __name__ == '__main__':
    main()
