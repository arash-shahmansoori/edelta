"""
Beta Tracking Analysis Script

Analyzes β (gate) values during inference to validate:
1. β spikes at correction tokens (Aha! moments)
2. β correlates with entropy (thermodynamic gating)
3. β is low during confident predictions (thermodynamic lock)

This validates the Illusion of Insight paper's observations
are mechanized in E∆-MHC-Geo.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import models
from proposed_model import GPT as EDeltaGPT, GPTConfig as EDeltaConfig


def load_model(checkpoint_path, device='cuda'):
    """Load a trained E∆-MHC-Geo model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    config = EDeltaConfig(**model_args)
    model = EDeltaGPT(config)
    
    state_dict = checkpoint['model']
    # Remove _orig_mod prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model


def get_beta_values(model, input_ids, device='cuda'):
    """
    Forward pass that extracts β values at each layer.
    
    Returns dict with:
    - layer_betas: list of (layer_idx, beta_tensor) 
    - positions: list of token positions
    """
    input_ids = input_ids.to(device)
    
    betas_per_layer = []
    
    # Hook to capture beta values
    def make_hook(layer_idx, is_attn=True):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                _, beta = output
                betas_per_layer.append({
                    'layer': layer_idx,
                    'type': 'attn' if is_attn else 'mlp',
                    'beta': beta.detach().cpu()
                })
        return hook
    
    # Register hooks on GeodesicDelta modules
    hooks = []
    for layer_idx, block in enumerate(model.transformer.h):
        if hasattr(block, 'geo_attn'):
            h = block.geo_attn.register_forward_hook(make_hook(layer_idx, True))
            hooks.append(h)
        if hasattr(block, 'geo_mlp'):
            h = block.geo_mlp.register_forward_hook(make_hook(layer_idx, False))
            hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return betas_per_layer, logits


def analyze_correction_task(model, samples, tokenizer_fn, device='cuda'):
    """
    Analyze β values on correction task samples.
    
    Expects samples with format:
    "Task: add 5 to X. X=10. Wait, I meant subtract 5. Answer:"
    
    Returns analysis of β values at correction tokens.
    """
    results = []
    
    correction_keywords = ['wait', 'actually', 'scratch', 'however', 'correction', 'ignore']
    
    for sample in samples:
        # Tokenize
        input_ids = tokenizer_fn(sample)
        if input_ids is None:
            continue
            
        input_ids = input_ids.unsqueeze(0).to(device)
        
        # Get beta values
        betas_per_layer, _ = get_beta_values(model, input_ids, device)
        
        # Find correction token position
        tokens = sample.lower().split()
        correction_pos = None
        for i, token in enumerate(tokens):
            for keyword in correction_keywords:
                if keyword in token:
                    correction_pos = i
                    break
            if correction_pos is not None:
                break
        
        if correction_pos is None:
            continue
        
        # Aggregate betas across layers
        all_betas = []
        for beta_info in betas_per_layer:
            beta = beta_info['beta']  # (1, S, 1) or similar
            if beta.dim() >= 2:
                beta = beta.squeeze()
                if beta.dim() == 1:
                    all_betas.append(beta.numpy())
        
        if len(all_betas) == 0:
            continue
            
        avg_beta = np.mean(all_betas, axis=0)
        
        # Compare beta at correction vs elsewhere
        if correction_pos < len(avg_beta):
            beta_at_correction = avg_beta[correction_pos:min(correction_pos+3, len(avg_beta))].mean()
            beta_elsewhere = avg_beta[:correction_pos].mean() if correction_pos > 0 else 0
            
            results.append({
                'sample': sample[:50] + '...',
                'correction_pos': correction_pos,
                'beta_at_correction': beta_at_correction,
                'beta_elsewhere': beta_elsewhere,
                'spike_ratio': beta_at_correction / (beta_elsewhere + 1e-6),
                'full_betas': avg_beta,
            })
    
    return results


def analyze_entropy_correlation(model, high_entropy_samples, low_entropy_samples, 
                                 tokenizer_fn, device='cuda'):
    """
    Analyze β values on entropy probe samples.
    
    Tests if β is higher for high-entropy (uncertain) samples
    and lower for low-entropy (confident) samples.
    """
    high_entropy_betas = []
    low_entropy_betas = []
    
    for sample in high_entropy_samples:
        input_ids = tokenizer_fn(sample)
        if input_ids is None:
            continue
        input_ids = input_ids.unsqueeze(0).to(device)
        
        betas_per_layer, _ = get_beta_values(model, input_ids, device)
        
        all_betas = [b['beta'].mean().item() for b in betas_per_layer]
        if all_betas:
            high_entropy_betas.append(np.mean(all_betas))
    
    for sample in low_entropy_samples:
        input_ids = tokenizer_fn(sample)
        if input_ids is None:
            continue
        input_ids = input_ids.unsqueeze(0).to(device)
        
        betas_per_layer, _ = get_beta_values(model, input_ids, device)
        
        all_betas = [b['beta'].mean().item() for b in betas_per_layer]
        if all_betas:
            low_entropy_betas.append(np.mean(all_betas))
    
    return {
        'high_entropy_beta_mean': np.mean(high_entropy_betas) if high_entropy_betas else 0,
        'high_entropy_beta_std': np.std(high_entropy_betas) if high_entropy_betas else 0,
        'low_entropy_beta_mean': np.mean(low_entropy_betas) if low_entropy_betas else 0,
        'low_entropy_beta_std': np.std(low_entropy_betas) if low_entropy_betas else 0,
        'separation': (np.mean(high_entropy_betas) - np.mean(low_entropy_betas)) 
                      if high_entropy_betas and low_entropy_betas else 0,
    }


def simple_tokenizer(text):
    """Simple character-level tokenizer."""
    try:
        return torch.tensor([ord(c) for c in text], dtype=torch.long)
    except:
        return None


def plot_beta_trace(beta_values, title="Beta Values Across Tokens", save_path=None):
    """Plot beta values across token positions."""
    plt.figure(figsize=(12, 4))
    plt.plot(beta_values, 'b-', linewidth=1.5)
    plt.xlabel('Token Position')
    plt.ylabel('β (Gate Value)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze beta tracking in E∆-MHC-Geo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--analysis', type=str, choices=['correction', 'entropy', 'both'], 
                        default='both', help='Type of analysis')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    
    if args.analysis in ['correction', 'both']:
        print("\n" + "="*60)
        print("CORRECTION TASK ANALYSIS (Aha! Moments)")
        print("="*60)
        
        # Sample correction tasks
        correction_samples = [
            "Task: add 5 to X. X=10. Wait, I meant subtract 5. Answer:",
            "Go North. Actually, no, go South. Direction:",
            "Set X=42. Scratch that, set X=17. Value:",
            "The answer is True. However, instead it's False. Final:",
        ]
        
        results = analyze_correction_task(model, correction_samples, simple_tokenizer, args.device)
        
        for r in results:
            print(f"\nSample: {r['sample']}")
            print(f"  β at correction: {r['beta_at_correction']:.4f}")
            print(f"  β elsewhere: {r['beta_elsewhere']:.4f}")
            print(f"  Spike ratio: {r['spike_ratio']:.2f}x")
        
        if results:
            avg_spike = np.mean([r['spike_ratio'] for r in results])
            print(f"\n✓ Average spike ratio: {avg_spike:.2f}x")
            if avg_spike > 1.5:
                print("  → β DOES spike at correction tokens! ✓")
            else:
                print("  → β does not show significant spike.")
    
    if args.analysis in ['entropy', 'both']:
        print("\n" + "="*60)
        print("ENTROPY CORRELATION ANALYSIS")
        print("="*60)
        
        high_entropy_samples = [
            "Pick any number: 1, 2, 3. I choose",
            "The color could be red or blue. It's",
            "Yes or no (your choice):",
        ]
        
        low_entropy_samples = [
            "2 + 2 =",
            "The capital of France is",
            "Water freezes at 0 degrees",
        ]
        
        results = analyze_entropy_correlation(
            model, high_entropy_samples, low_entropy_samples, 
            simple_tokenizer, args.device
        )
        
        print(f"\nHigh entropy samples:")
        print(f"  β mean: {results['high_entropy_beta_mean']:.4f} ± {results['high_entropy_beta_std']:.4f}")
        print(f"\nLow entropy samples:")
        print(f"  β mean: {results['low_entropy_beta_mean']:.4f} ± {results['low_entropy_beta_std']:.4f}")
        print(f"\nSeparation: {results['separation']:.4f}")
        
        if results['separation'] > 0:
            print("  → β IS higher for high-entropy samples! ✓")
        else:
            print("  → β correlation is inverted or absent.")


if __name__ == "__main__":
    main()
