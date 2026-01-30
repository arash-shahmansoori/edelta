"""
Diagnostic script to analyze the Geodesic-Delta mechanism.
Loads a trained checkpoint and examines:
1. The learned u, v rotation generators
2. The actual rotation magnitude ||Qx - x||
3. The beta values across different inputs
4. Whether Q is actually orthogonal (sanity check)
"""

import torch
import numpy as np
import os
from proposed_model import GPT, GPTConfig, GeodesicDelta

def load_checkpoint(ckpt_path):
    """Load a trained checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    
    # Add geodesic defaults if missing
    model_args.setdefault('use_damper', True)
    model_args.setdefault('use_static_gate', False)
    model_args.setdefault('geo_lr_mult', 50.0)
    
    config = GPTConfig(**model_args)
    model = GPT(config)
    
    # Handle compiled model keys
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    return model, config

def analyze_rotation_generators(model):
    """Analyze the learned u, v vectors."""
    print("\n" + "="*60)
    print("ROTATION GENERATOR ANALYSIS")
    print("="*60)
    
    for layer_idx, block in enumerate(model.transformer.h):
        print(f"\n--- Layer {layer_idx} ---")
        
        for name, geo in [("Attention", block.geo_attn), ("MLP", block.geo_mlp)]:
            u = geo.u.detach().squeeze()  # (n_streams,)
            v = geo.v.detach().squeeze()  # (n_streams,)
            
            # Compute the skew-symmetric generator A = uv^T - vu^T
            A = torch.outer(u, v) - torch.outer(v, u)
            
            # Eigenvalues of A (should be purely imaginary for skew-symmetric)
            eigenvalues = torch.linalg.eigvals(A)
            
            # Frobenius norm of A (rotation "strength")
            A_norm = torch.norm(A, 'fro')
            
            # Angle between u and v (determines rotation plane)
            cos_angle = torch.dot(u, v) / (torch.norm(u) * torch.norm(v) + 1e-8)
            angle_deg = torch.acos(cos_angle.clamp(-1, 1)) * 180 / np.pi
            
            print(f"\n  {name} GeodesicDelta:")
            print(f"    ||u||: {torch.norm(u):.6f}, ||v||: {torch.norm(v):.6f}")
            print(f"    Angle(u,v): {angle_deg:.2f}°")
            print(f"    ||A||_F (rotation strength): {A_norm:.6f}")
            print(f"    A eigenvalues (should be ±imaginary): {eigenvalues.numpy()}")
            
            if hasattr(geo, 'w_alpha'):
                print(f"    w_alpha: {geo.w_alpha.item():.6f}")
                print(f"    b_init: {geo.b_init.item():.6f}")
            elif hasattr(geo, 'static_beta'):
                print(f"    static_beta: {geo.static_beta.item():.6f}")

def analyze_rotation_magnitude(model, device='cuda'):
    """Measure how much rotation actually happens on random inputs."""
    print("\n" + "="*60)
    print("ROTATION MAGNITUDE ANALYSIS")
    print("="*60)
    
    model.eval()
    model.to(device)
    
    # Create random input
    batch_size, seq_len = 4, 64
    vocab_size = model.config.vocab_size
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        # Get embeddings
        tok_emb = model.transformer.wte(x)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        h = model.transformer.drop(tok_emb + pos_emb)
        
        # Analyze each layer
        for layer_idx, block in enumerate(model.transformer.h):
            print(f"\n--- Layer {layer_idx} ---")
            
            # Get rotation output
            h_rotated, beta = block.geo_attn(h, return_beta=True)
            
            # Measure rotation magnitude
            rotation_diff = h_rotated - h
            rotation_magnitude = torch.norm(rotation_diff, dim=-1).mean()
            relative_rotation = rotation_magnitude / torch.norm(h, dim=-1).mean()
            
            # Beta statistics
            beta_mean = beta.mean().item()
            beta_max = beta.max().item()
            
            print(f"  Attention GeodesicDelta:")
            print(f"    β mean: {beta_mean:.6f}, β max: {beta_max:.6f}")
            print(f"    ||Qx - x|| (absolute): {rotation_magnitude:.6f}")
            print(f"    ||Qx - x|| / ||x|| (relative): {relative_rotation:.4%}")
            
            # Continue through the block
            h = block(h)

def verify_orthogonality(model, device='cuda'):
    """Verify that Q is actually orthogonal (Q^T Q = I)."""
    print("\n" + "="*60)
    print("ORTHOGONALITY VERIFICATION")
    print("="*60)
    
    model.eval()
    model.to(device)
    
    for layer_idx, block in enumerate(model.transformer.h):
        print(f"\n--- Layer {layer_idx} ---")
        
        for name, geo in [("Attention", block.geo_attn), ("MLP", block.geo_mlp)]:
            # Compute Q for a fixed beta
            u = geo.u.to(device)
            v = geo.v.to(device)
            
            A = torch.matmul(u, v.transpose(-1, -2)) - torch.matmul(v, u.transpose(-1, -2))
            
            # Test multiple beta values
            for beta_val in [0.001, 0.01, 0.1, 1.0]:
                M = beta_val * A
                n = geo.n_streams
                I = torch.eye(n, device=device).view(1, 1, n, n)
                
                Q = torch.linalg.solve(I + M, I - M)
                
                # Check Q^T Q = I
                QtQ = torch.matmul(Q.transpose(-1, -2), Q)
                error = torch.norm(QtQ - I, 'fro').item()
                
                print(f"  {name} @ β={beta_val}: ||Q^T Q - I||_F = {error:.2e}")

def analyze_purity_proxy(model, device='cuda'):
    """Analyze the purity proxy values on different inputs."""
    print("\n" + "="*60)
    print("PURITY PROXY ANALYSIS")
    print("="*60)
    
    model.eval()
    model.to(device)
    
    # Create inputs with different "entropy" levels
    batch_size, seq_len = 4, 64
    vocab_size = model.config.vocab_size
    
    # Random input (high entropy)
    x_random = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Repeated token (low entropy)
    x_repeat = torch.full((batch_size, seq_len), 42, dtype=torch.long, device=device)
    
    # Sequential tokens (medium entropy)
    x_seq = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    with torch.no_grad():
        for name, x in [("Random", x_random), ("Repeated", x_repeat), ("Sequential", x_seq)]:
            tok_emb = model.transformer.wte(x)
            pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
            pos_emb = model.transformer.wpe(pos)
            h = model.transformer.drop(tok_emb + pos_emb)
            
            # Get purity for first layer
            geo = model.transformer.h[0].geo_attn
            n_streams = geo.n_streams
            d_stream = geo.d_stream
            x_streams = h.view(batch_size, seq_len, n_streams, d_stream)
            
            phi = geo.get_purity_proxy(x_streams)
            
            print(f"\n  {name} input:")
            print(f"    Φ (purity proxy) mean: {phi.mean().item():.6f}")
            print(f"    Φ range: [{phi.min().item():.6f}, {phi.max().item():.6f}]")
            
            if hasattr(geo, 'w_alpha'):
                beta = torch.nn.functional.softplus(geo.w_alpha * phi + geo.b_init)
                print(f"    β mean: {beta.mean().item():.6f}")

def main():
    import sys
    
    # Default checkpoint path
    ckpt_paths = [
        'out-grok-boosted/ckpt.pt',
        'out-reverse-geodesic/ckpt.pt',
        'out-grok-geodesic/ckpt.pt',
    ]
    
    # Use command line arg if provided
    if len(sys.argv) > 1:
        ckpt_paths = [sys.argv[1]]
    
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue
        
        print("\n" + "#"*70)
        print(f"# ANALYZING: {ckpt_path}")
        print("#"*70)
        
        model, config = load_checkpoint(ckpt_path)
        
        analyze_rotation_generators(model)
        verify_orthogonality(model)
        
        if torch.cuda.is_available():
            analyze_rotation_magnitude(model, device='cuda')
            analyze_purity_proxy(model, device='cuda')
        else:
            print("\nSkipping GPU-based analysis (no CUDA available)")

if __name__ == '__main__':
    main()
