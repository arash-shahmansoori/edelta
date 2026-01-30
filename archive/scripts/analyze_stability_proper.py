"""
Proper Stability Analysis

The original stability test was flawed because it used single-vector autoregression,
which doesn't match how models were trained.

This script tests stability DURING SEQUENCE PROCESSING:
1. Feed a sequence of unit vectors
2. Check if output vectors maintain unit norm
3. This matches training and tests the actual model behavior
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'GPT': '#4285F4',
    'DDL': '#EA4335',
    'mHC': '#FBBC05',
    'E∆-MHC-Geo': '#34A853',
}


def load_model_from_checkpoint(ckpt_path, device='cuda'):
    """Load a model from checkpoint with correct configuration."""
    from model import GPT as BaselineGPT, GPTConfig as BaselineConfig
    from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
    from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
    from proposed_model_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig
    from train_continuous import ContinuousModelWrapper
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = checkpoint['config']
    model_type = saved_config['model_type']
    
    for k, v in checkpoint['model'].items():
        if 'pos_emb' in k:
            block_size = v.shape[1]
            break
    
    model_map = {
        'gpt2': (BaselineGPT, BaselineConfig),
        'ddl': (DDLGPT, DDLConfig),
        'mhc': (mHCGPT, mHCConfig),
        'edelta': (EdeltaGPT, EdeltaConfig),
    }
    
    GPTClass, ConfigClass = model_map[model_type]
    
    config_args = {
        'n_layer': saved_config['n_layer'],
        'n_head': saved_config['n_head'],
        'n_embd': saved_config['n_embd'],
        'dropout': saved_config['dropout'],
        'bias': False,
        'block_size': block_size,
        'vocab_size': 1
    }
    if model_type in ['mhc', 'edelta']:
        config_args['n_streams'] = saved_config.get('n_streams', 4)
    
    config = ConfigClass(**config_args)
    core_model = GPTClass(config)
    
    dataset = saved_config['dataset']
    input_dims = {'gyroscope': 16, 'correction': 32, 'stability': 64}
    input_dim = input_dims.get(dataset, 64)
    
    model = ContinuousModelWrapper(core_model, config, input_dim, block_size)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model, saved_config


def test_sequence_norm_preservation(model, seq_len=100, dim=64, n_sequences=50, device='cuda'):
    """
    PROPER STABILITY TEST: Feed sequences of unit vectors, check output norms.
    
    This tests: Does the model preserve norms during normal sequence processing?
    """
    model.eval()
    
    all_input_norms = []
    all_output_norms = []
    all_mse = []
    
    with torch.no_grad():
        for _ in range(n_sequences):
            # Create sequence of random unit vectors
            x = torch.randn(1, seq_len, dim, device=device)
            x = F.normalize(x, dim=-1)  # Make unit vectors
            
            # Target is same as input (identity mapping)
            y = x.clone()
            
            # Forward pass
            pred, loss = model(x, y)
            
            # Compute norms
            input_norms = torch.norm(x, dim=-1).squeeze()  # (seq_len,)
            output_norms = torch.norm(pred, dim=-1).squeeze()  # (seq_len,)
            
            all_input_norms.append(input_norms.cpu().numpy())
            all_output_norms.append(output_norms.cpu().numpy())
            all_mse.append(loss.item() if loss is not None else 0)
    
    return {
        'input_norms': np.array(all_input_norms),   # (n_seq, seq_len)
        'output_norms': np.array(all_output_norms), # (n_seq, seq_len)
        'mse': np.mean(all_mse),
    }


def test_long_range_preservation(model, key_pos=10, query_pos=100, dim=64, n_tests=50, device='cuda'):
    """
    Test if model can preserve a key vector over long distance.
    
    Sequence: [noise, noise, ..., KEY, noise, noise, ..., QUERY]
    Target at QUERY position should be KEY
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for _ in range(n_tests):
            seq_len = query_pos + 1
            
            # Create noise sequence
            x = torch.randn(1, seq_len, dim, device=device)
            x = F.normalize(x, dim=-1)
            
            # Insert key at key_pos
            key = torch.randn(1, 1, dim, device=device)
            key = F.normalize(key, dim=-1)
            x[:, key_pos, :] = key.squeeze(1)
            
            # Target: key at query position
            y = x.clone()
            y[:, query_pos, :] = key.squeeze(1)
            
            # Forward pass
            pred, _ = model(x, y)
            
            # Check if output at query_pos matches key
            output_at_query = pred[:, query_pos, :]
            
            cosine_sim = F.cosine_similarity(output_at_query, key.squeeze(1), dim=-1).item()
            output_norm = torch.norm(output_at_query, dim=-1).item()
            mse = F.mse_loss(output_at_query, key.squeeze(1)).item()
            
            results.append({
                'cosine_sim': cosine_sim,
                'output_norm': output_norm,
                'mse': mse,
            })
    
    return {
        'cosine_sim_mean': np.mean([r['cosine_sim'] for r in results]),
        'cosine_sim_std': np.std([r['cosine_sim'] for r in results]),
        'output_norm_mean': np.mean([r['output_norm'] for r in results]),
        'output_norm_std': np.std([r['output_norm'] for r in results]),
        'mse_mean': np.mean([r['mse'] for r in results]),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dirs = {
        'GPT': 'out-stability-baseline',
        'DDL': 'out-stability-ddl',
        'mHC': 'out-stability-mhc',
        'E∆-MHC-Geo': 'out-stability-proposed',
    }
    
    print("=" * 70)
    print("PROPER STABILITY ANALYSIS")
    print("=" * 70)
    print()
    
    # Test 1: Sequence norm preservation
    print("Test 1: Sequence Norm Preservation")
    print("-" * 50)
    print("Feed sequences of unit vectors, check if output norms stay at 1.0")
    print()
    
    seq_results = {}
    for model_name, ckpt_dir in model_dirs.items():
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {model_name} - no checkpoint")
            continue
        
        print(f"  Testing {model_name}...")
        model, config = load_model_from_checkpoint(ckpt_path, device)
        
        result = test_sequence_norm_preservation(model, seq_len=100, dim=64, device=device)
        seq_results[model_name] = result
        
        mean_output_norm = result['output_norms'].mean()
        std_output_norm = result['output_norms'].std()
        
        print(f"    Mean output norm: {mean_output_norm:.4f} ± {std_output_norm:.4f}")
        print(f"    MSE loss: {result['mse']:.6f}")
        
        del model
        torch.cuda.empty_cache()
    
    print()
    
    # Test 2: Long-range preservation
    print("Test 2: Long-Range Key Preservation")
    print("-" * 50)
    print("Can model preserve a key vector over 90 steps of noise?")
    print()
    
    lr_results = {}
    for model_name, ckpt_dir in model_dirs.items():
        ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            continue
        
        print(f"  Testing {model_name}...")
        model, config = load_model_from_checkpoint(ckpt_path, device)
        
        result = test_long_range_preservation(model, key_pos=10, query_pos=100, dim=64, device=device)
        lr_results[model_name] = result
        
        print(f"    Cosine similarity: {result['cosine_sim_mean']:.4f} ± {result['cosine_sim_std']:.4f}")
        print(f"    Output norm: {result['output_norm_mean']:.4f} ± {result['output_norm_std']:.4f}")
        print(f"    MSE: {result['mse_mean']:.6f}")
        
        del model
        torch.cuda.empty_cache()
    
    # Create visualization
    print()
    print("Creating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Plot 1: Output norm distribution across sequence
    ax1 = axes[0]
    for model_name, result in seq_results.items():
        norms = result['output_norms']  # (n_seq, seq_len)
        mean_norm = norms.mean(axis=0)  # Average across sequences
        std_norm = norms.std(axis=0)
        steps = np.arange(len(mean_norm))
        
        ax1.plot(steps, mean_norm, color=COLORS[model_name], label=model_name)
        ax1.fill_between(steps, mean_norm - std_norm, mean_norm + std_norm,
                        color=COLORS[model_name], alpha=0.2)
    
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Target ||v||=1')
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Output Norm ||pred||')
    ax1.set_title('(a) Norm During Sequence Processing')
    ax1.legend(loc='best')
    ax1.set_ylim(0.5, 1.5)
    
    # Plot 2: Norm deviation bar chart
    ax2 = axes[1]
    models = list(seq_results.keys())
    deviations = [np.abs(seq_results[m]['output_norms'] - 1.0).mean() for m in models]
    colors = [COLORS[m] for m in models]
    
    bars = ax2.bar(models, deviations, color=colors, edgecolor='black')
    ax2.set_ylabel('Mean |Norm - 1.0|')
    ax2.set_title('(b) Average Norm Deviation')
    ax2.set_xticklabels(models, rotation=15)
    
    for bar, val in zip(bars, deviations):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Long-range preservation
    ax3 = axes[2]
    models = list(lr_results.keys())
    cosines = [lr_results[m]['cosine_sim_mean'] for m in models]
    cosine_stds = [lr_results[m]['cosine_sim_std'] for m in models]
    colors = [COLORS[m] for m in models]
    
    bars = ax3.bar(models, cosines, color=colors, edgecolor='black', yerr=cosine_stds, capsize=5)
    ax3.set_ylabel('Cosine Similarity (retrieved vs original)')
    ax3.set_title('(c) Long-Range Key Retrieval')
    ax3.set_xticklabels(models, rotation=15)
    ax3.set_ylim(0, 1.1)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, cosines):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    
    fig.suptitle('Proper Stability Analysis (Sequence-Based)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_stability_proper.png', dpi=300, bbox_inches='tight')
    print("Saved: fig_stability_proper.png")
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Test 1: Sequence Norm Preservation (lower deviation = better)")
    print("-" * 50)
    for model_name in seq_results:
        dev = np.abs(seq_results[model_name]['output_norms'] - 1.0).mean()
        print(f"  {model_name:<15}: deviation = {dev:.4f}")
    
    print()
    print("Test 2: Long-Range Key Retrieval (higher cosine = better)")
    print("-" * 50)
    for model_name in lr_results:
        cos = lr_results[model_name]['cosine_sim_mean']
        print(f"  {model_name:<15}: cosine = {cos:.4f}")
    
    print()
    print("INTERPRETATION:")
    print("  - These tests measure ACTUAL model behavior during sequence processing")
    print("  - Unlike the flawed autoregressive test, this matches training conditions")
    print("  - Norm deviation shows how well the model preserves vector magnitudes")
    print("  - Long-range retrieval tests if information is preserved over distance")


if __name__ == '__main__':
    main()
