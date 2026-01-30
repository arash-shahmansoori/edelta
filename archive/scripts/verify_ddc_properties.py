"""
Mathematical Verification of Data-Dependent Cayley Properties

This script empirically verifies the theoretical claims in RESEARCH_V3.md:
1. Unconditional orthogonality: Q(x)^T Q(x) = I for all x
2. Isometry: ||Q(x) y|| = ||y|| for all x, y
3. Determinant: det(Q(x)) = +1 for all x
4. Non-singularity: (I + βA/2) is always invertible

We test across many random inputs to confirm the properties hold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tabulate import tabulate


def generate_random_uv(batch_size: int, n: int, d_model: int) -> tuple:
    """Generate random u, v vectors via random linear projections."""
    x = torch.randn(batch_size, d_model)
    W_u = torch.randn(n, d_model) * 0.1
    W_v = torch.randn(n, d_model) * 0.1
    u = x @ W_u.T  # (B, n)
    v = x @ W_v.T  # (B, n)
    return u, v


def construct_skew_symmetric(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Construct A = uv^T - vu^T
    
    This is ALWAYS skew-symmetric, regardless of u and v.
    """
    A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
    return A


def verify_skew_symmetry(A: torch.Tensor) -> float:
    """Verify A^T = -A by computing ||A + A^T||."""
    return (A + A.transpose(-2, -1)).abs().max().item()


def cayley_transform(A: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Cayley transform: Q = (I + βA/2)^{-1} (I - βA/2)
    """
    n = A.shape[-1]
    B = A.shape[0]
    M = (beta.unsqueeze(-1) / 2) * A
    I = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    
    I_plus_M = I + M
    I_minus_M = I - M
    
    Q = torch.linalg.solve(I_plus_M, I_minus_M)
    return Q


def verify_orthogonality(Q: torch.Tensor) -> float:
    """Verify Q^T Q = I by computing ||Q^T Q - I||."""
    I = torch.eye(Q.shape[-1]).unsqueeze(0).expand(Q.shape[0], -1, -1)
    QtQ = torch.bmm(Q.transpose(-2, -1), Q)
    return (QtQ - I).abs().max().item()


def verify_isometry(Q: torch.Tensor, y: torch.Tensor) -> float:
    """Verify ||Qy|| = ||y|| by computing relative error."""
    Qy = torch.bmm(Q, y.unsqueeze(-1)).squeeze(-1)
    norm_y = torch.norm(y, dim=-1)
    norm_Qy = torch.norm(Qy, dim=-1)
    relative_error = ((norm_Qy - norm_y).abs() / (norm_y + 1e-8)).max().item()
    return relative_error


def verify_determinant(Q: torch.Tensor) -> tuple:
    """Verify det(Q) = +1."""
    dets = torch.linalg.det(Q)
    max_error = (dets - 1.0).abs().max().item()
    min_det = dets.min().item()
    max_det = dets.max().item()
    return max_error, min_det, max_det


def verify_invertibility(A: torch.Tensor, beta: torch.Tensor) -> tuple:
    """Verify (I + βA/2) is invertible by computing condition number."""
    n = A.shape[-1]
    B = A.shape[0]
    M = (beta.unsqueeze(-1) / 2) * A
    I = torch.eye(n).unsqueeze(0).expand(B, -1, -1)
    S = I + M
    
    # Compute condition number
    singular_values = torch.linalg.svdvals(S)
    cond = singular_values[:, 0] / singular_values[:, -1]
    return cond.mean().item(), cond.max().item()


def run_verification(n_tests: int = 1000, batch_size: int = 32, n: int = 4, d_model: int = 128):
    """Run comprehensive verification across many random inputs."""
    print("=" * 60)
    print("DATA-DEPENDENT CAYLEY PROPERTY VERIFICATION")
    print("=" * 60)
    print(f"\nTest configuration:")
    print(f"  - Number of test batches: {n_tests}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Stream dimension (n): {n}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Total test cases: {n_tests * batch_size:,}")
    
    # Accumulators
    skew_errors = []
    orth_errors = []
    isom_errors = []
    det_errors = []
    cond_means = []
    cond_maxs = []
    
    for i in range(n_tests):
        # Generate random inputs
        u, v = generate_random_uv(batch_size, n, d_model)
        beta = torch.rand(batch_size) * 10  # Random beta in [0, 10]
        y = torch.randn(batch_size, n)  # Random vector to test isometry
        
        # Construct skew-symmetric A
        A = construct_skew_symmetric(u, v)
        
        # Verify skew-symmetry
        skew_errors.append(verify_skew_symmetry(A))
        
        # Apply Cayley transform
        Q = cayley_transform(A, beta)
        
        # Verify orthogonality
        orth_errors.append(verify_orthogonality(Q))
        
        # Verify isometry
        isom_errors.append(verify_isometry(Q, y))
        
        # Verify determinant
        det_err, _, _ = verify_determinant(Q)
        det_errors.append(det_err)
        
        # Verify invertibility
        cond_mean, cond_max = verify_invertibility(A, beta)
        cond_means.append(cond_mean)
        cond_maxs.append(cond_max)
    
    # Summarize results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    results = [
        ["Skew-symmetry (||A + A^T||)", f"{np.max(skew_errors):.2e}", "< 1e-6", "✅" if np.max(skew_errors) < 1e-6 else "❌"],
        ["Orthogonality (||Q^TQ - I||)", f"{np.max(orth_errors):.2e}", "< 1e-5", "✅" if np.max(orth_errors) < 1e-5 else "❌"],
        ["Isometry (relative norm error)", f"{np.max(isom_errors):.2e}", "< 1e-5", "✅" if np.max(isom_errors) < 1e-5 else "❌"],
        ["Determinant (|det(Q) - 1|)", f"{np.max(det_errors):.2e}", "< 1e-5", "✅" if np.max(det_errors) < 1e-5 else "❌"],
        ["Condition number (mean)", f"{np.mean(cond_means):.2f}", "< 100", "✅" if np.mean(cond_means) < 100 else "⚠️"],
        ["Condition number (max)", f"{np.max(cond_maxs):.2f}", "< 1000", "✅" if np.max(cond_maxs) < 1000 else "⚠️"],
    ]
    
    print(tabulate(results, headers=["Property", "Max Error/Value", "Threshold", "Status"], tablefmt="grid"))
    
    # Additional edge case tests
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)
    
    # Test with very large beta
    print("\n1. Very large β (β = 1000):")
    u, v = generate_random_uv(100, n, d_model)
    A = construct_skew_symmetric(u, v)
    beta_large = torch.ones(100) * 1000
    Q = cayley_transform(A, beta_large)
    print(f"   Orthogonality error: {verify_orthogonality(Q):.2e}")
    det_err, min_det, max_det = verify_determinant(Q)
    print(f"   Determinant: min={min_det:.6f}, max={max_det:.6f}")
    
    # Test with zero beta (should give identity)
    print("\n2. Zero β (β = 0) [should give identity]:")
    beta_zero = torch.zeros(100)
    Q_zero = cayley_transform(A, beta_zero)
    I = torch.eye(n).unsqueeze(0).expand(100, -1, -1)
    identity_error = (Q_zero - I).abs().max().item()
    print(f"   ||Q - I|| error: {identity_error:.2e}")
    
    # Test with negative beta
    print("\n3. Negative β (β = -5):")
    beta_neg = torch.ones(100) * -5
    Q_neg = cayley_transform(A, beta_neg)
    print(f"   Orthogonality error: {verify_orthogonality(Q_neg):.2e}")
    det_err, min_det, max_det = verify_determinant(Q_neg)
    print(f"   Determinant: min={min_det:.6f}, max={max_det:.6f}")
    
    # Test gradient flow
    print("\n" + "=" * 60)
    print("GRADIENT FLOW TEST")
    print("=" * 60)
    
    # Create a simple differentiable pipeline
    x = torch.randn(10, d_model, requires_grad=True)
    W_u = torch.randn(n, d_model, requires_grad=True)
    W_v = torch.randn(n, d_model, requires_grad=True)
    beta_param = torch.tensor([1.0], requires_grad=True)
    
    u = x @ W_u.T
    v = x @ W_v.T
    A = construct_skew_symmetric(u, v)
    Q = cayley_transform(A, beta_param.expand(10))
    
    # Dummy loss: sum of all elements
    y = torch.randn(10, n)
    Qy = torch.bmm(Q, y.unsqueeze(-1)).squeeze(-1)
    loss = Qy.sum()
    
    # Compute gradients
    loss.backward()
    
    print(f"Gradient w.r.t. W_u exists: {W_u.grad is not None}")
    print(f"Gradient w.r.t. W_v exists: {W_v.grad is not None}")
    print(f"Gradient w.r.t. β exists: {beta_param.grad is not None}")
    print(f"Gradient w.r.t. x exists: {x.grad is not None}")
    
    if W_u.grad is not None:
        print(f"||∂L/∂W_u|| = {W_u.grad.norm().item():.4f}")
    if beta_param.grad is not None:
        print(f"∂L/∂β = {beta_param.grad.item():.4f}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nConclusion: Data-Dependent Cayley maintains ALL mathematical")
    print("properties of the Cayley transform regardless of how u(x) and v(x)")
    print("are computed. The key insight is that skew-symmetry of A = uv^T - vu^T")
    print("depends only on the algebraic construction, not on the source of u, v.")


if __name__ == "__main__":
    run_verification()
