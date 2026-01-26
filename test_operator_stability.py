"""
TRUE TEST: Operator Stability Under Repeated Application

This test DIRECTLY measures what DDC claims to excel at:
- Apply the learned operator Q repeatedly: x, Q(x), Q(Q(x)), Q(Q(Q(x))), ...
- Measure norm drift over 100+ iterations
- DDC should have ZERO drift (unconditional orthogonality)
- DDL should have drift (only orthogonal at β=2)

This is the DEFINITIVE test of DDC's theoretical advantage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class DDLOperator(nn.Module):
    """Householder reflection operator (DDL style)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.k_raw = nn.Parameter(torch.randn(dim))
        self.beta_raw = nn.Parameter(torch.zeros(1))  # Will learn β
    
    def forward(self, x):
        # x: (B, D)
        k = F.normalize(self.k_raw, dim=0)
        beta = 2 * torch.sigmoid(self.beta_raw)  # β ∈ (0, 2)
        
        # H(x) = x - β * k * (k^T x)
        dot = (x * k).sum(dim=-1, keepdim=True)
        return x - beta * dot * k


class DDCOperator(nn.Module):
    """Data-Dependent Cayley operator (UNCONDITIONAL orthogonality)."""
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.d_stream = dim // n_streams
        
        # Fixed u, v for this test (can also be data-dependent)
        self.u = nn.Parameter(torch.randn(n_streams) * 0.1)
        self.v = nn.Parameter(torch.randn(n_streams) * 0.1)
        self.beta_raw = nn.Parameter(torch.zeros(1))
        
        self.register_buffer('I', torch.eye(n_streams))
    
    def forward(self, x):
        # x: (B, D)
        B = x.shape[0]
        
        # Skew-symmetric A = uv^T - vu^T (ALWAYS skew-symmetric!)
        A = torch.outer(self.u, self.v) - torch.outer(self.v, self.u)
        
        # Cayley transform: Q = (I + βA/2)^{-1} (I - βA/2)
        beta = F.softplus(self.beta_raw)
        M = (beta / 2) * A
        Q = torch.linalg.solve(self.I + M, self.I - M)  # GUARANTEED orthogonal!
        
        # Apply to streams
        x_streams = x.view(B, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('ij,bjd->bid', Q, x_streams)
        
        return x_rotated.reshape(B, self.dim)


def measure_stability(operator, n_steps=100, batch_size=100, dim=16):
    """
    Apply operator repeatedly and measure norm drift.
    
    Returns: list of mean norms at each step
    """
    # Start with unit vectors
    x = torch.randn(batch_size, dim)
    x = x / x.norm(dim=-1, keepdim=True)
    
    norms = [x.norm(dim=-1).mean().item()]
    
    with torch.no_grad():
        for _ in range(n_steps):
            x = operator(x)
            norms.append(x.norm(dim=-1).mean().item())
    
    return norms


def run_stability_test():
    """Run the definitive stability test."""
    print("=" * 70)
    print("DEFINITIVE TEST: Operator Stability Under Repeated Application")
    print("=" * 70)
    
    dim = 16
    n_steps = 200
    batch_size = 100
    
    # Create operators
    ddl_op = DDLOperator(dim)
    ddc_op = DDCOperator(dim, n_streams=4)
    
    # Initialize with random non-trivial parameters
    with torch.no_grad():
        ddl_op.beta_raw.fill_(0.5)  # β ≈ 1.2 (not exactly 2!)
        ddc_op.beta_raw.fill_(1.0)  # β ≈ 1.3
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dim}")
    print(f"  Steps: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  DDL β: {2 * torch.sigmoid(ddl_op.beta_raw).item():.4f} (should be ≠ 2 for drift)")
    print(f"  DDC β: {F.softplus(ddc_op.beta_raw).item():.4f}")
    
    # Measure stability
    print("\nMeasuring stability...")
    ddl_norms = measure_stability(ddl_op, n_steps, batch_size, dim)
    ddc_norms = measure_stability(ddc_op, n_steps, batch_size, dim)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Norm after N iterations (started at 1.0)")
    print("=" * 70)
    print(f"{'Step':<10} {'DDL Norm':<15} {'DDC Norm':<15} {'DDL Drift':<15} {'DDC Drift':<15}")
    print("-" * 70)
    
    for step in [0, 10, 50, 100, 150, 200]:
        if step < len(ddl_norms):
            ddl_drift = abs(ddl_norms[step] - 1.0)
            ddc_drift = abs(ddc_norms[step] - 1.0)
            print(f"{step:<10} {ddl_norms[step]:<15.6f} {ddc_norms[step]:<15.6f} {ddl_drift:<15.6f} {ddc_drift:<15.6f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    ddl_final_drift = abs(ddl_norms[-1] - 1.0)
    ddc_final_drift = abs(ddc_norms[-1] - 1.0)
    
    print(f"\nFinal norm drift after {n_steps} iterations:")
    print(f"  DDL: {ddl_final_drift:.6f} ({ddl_final_drift * 100:.2f}% drift)")
    print(f"  DDC: {ddc_final_drift:.6f} ({ddc_final_drift * 100:.4f}% drift)")
    
    if ddl_final_drift > 0.01:
        print("\n⚠️  DDL shows significant norm drift (as expected, since β ≠ 2)")
    else:
        print("\n✓  DDL has minimal drift (β is close to 2)")
    
    if ddc_final_drift < 1e-5:
        print("✅ DDC has ZERO drift (unconditional orthogonality CONFIRMED)")
    else:
        print(f"⚠️  DDC has unexpected drift: {ddc_final_drift}")
    
    # Test with DDL at β exactly 2
    print("\n" + "=" * 70)
    print("CONTROL TEST: DDL with β = 2 (should also have zero drift)")
    print("=" * 70)
    
    with torch.no_grad():
        # Set β to exactly 2 by solving sigmoid(x) = 1, i.e., x → ∞
        ddl_op.beta_raw.fill_(10.0)  # sigmoid(10) ≈ 0.99995, so β ≈ 1.9999
    
    print(f"  DDL β: {2 * torch.sigmoid(ddl_op.beta_raw).item():.6f}")
    
    ddl_norms_at_2 = measure_stability(ddl_op, n_steps, batch_size, dim)
    ddl_drift_at_2 = abs(ddl_norms_at_2[-1] - 1.0)
    
    print(f"  DDL drift at β≈2: {ddl_drift_at_2:.6f}")
    
    if ddl_drift_at_2 < 1e-4:
        print("  ✓ DDL with β=2 also has minimal drift (orthogonal at this point)")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ddl_norms, label=f'DDL (β≈1.2)', alpha=0.8)
    plt.plot(ddc_norms, label='DDC (any β)', alpha=0.8)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Ideal (norm=1)')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Norm')
    plt.title('Norm Stability: DDL vs DDC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy([abs(n - 1.0) + 1e-10 for n in ddl_norms], label='DDL drift')
    plt.semilogy([abs(n - 1.0) + 1e-10 for n in ddc_norms], label='DDC drift')
    plt.xlabel('Iteration')
    plt.ylabel('Norm Drift (log scale)')
    plt.title('Drift Accumulation (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('operator_stability_test.png', dpi=150)
    print(f"\nPlot saved to: operator_stability_test.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
KEY FINDING:
- DDC's Cayley transform is UNCONDITIONALLY orthogonal
- DDL's Householder is only orthogonal at β = 2
- When β ≠ 2, DDL accumulates norm drift over iterations

This is the FUNDAMENTAL difference between DDC and DDL:
- DDC: Q^T Q = I for ANY β (guaranteed by math)
- DDL: H^T H = I ONLY when β ∈ {0, 2}

For tasks requiring repeated application of a transformation
(e.g., recurrent processing, iterative refinement),
DDC provides guaranteed stability while DDL does not.
""")


if __name__ == '__main__':
    run_stability_test()
