#!/usr/bin/env python3
"""
Architectural Correction Dataset - Designed to Expose Model Differences

============================================================================
Key Insight from "The Illusion of Insight" (arXiv:2601.00514v1):
============================================================================
The paper shows that "Aha!" moments are about INSTANT belief flips, not gradual
corrections. This maps directly to our geometric framework:

- ROTATIONS (det=+1): Gradual, continuous transformations
  - Like step-by-step reasoning refinement
  - Cayley transform can do this perfectly
  
- REFLECTIONS (det=-1): Instant, discontinuous flips  
  - Like sudden "Wait, that's completely wrong!" moments
  - Householder reflection can do this perfectly
  - Cayley CANNOT do reflections (topological constraint)

============================================================================
Model Comparison (arXiv:2601.00417 for DDL):
============================================================================

| Model  | Operator                            | det(A) | Can Reflect? |
|--------|-------------------------------------|--------|--------------|
| GPT2   | x + f(x) (standard residual)        | N/A    | No inductive bias |
| DDL    | A = I - β·kk^T (rank-1 perturbation)| 1-β    | YES at β=2   |
| Cayley | Q = (I+M)^{-1}(I-M)                 | +1     | NO           |
| E∆     | β·Cayley + (1-β)·Householder        | varies | YES          |

DDL (Deep Delta Learning, arXiv:2601.00417):
- Uses rank-1 perturbation of identity
- β ∈ [0, 2] controls: identity (0) → projection (1) → reflection (2)
- det(I - β·kk^T) = 1 - β, so at β=2 we get det=-1 (reflection)

Cayley (Pure Rotation):
- Q = (I+M)^{-1}(I-M) where M is skew-symmetric
- Mathematically GUARANTEED det(Q) = +1
- CANNOT produce reflections - topological constraint

============================================================================
Dataset Design Philosophy:
============================================================================
We create 4 tasks that systematically test architectural capabilities:

1. PURE_ROTATION: Only rotations needed (All rotation-capable models should excel)
2. PURE_REFLECTION: Only reflections needed (DDL and E∆ should work, Cayley FAILS)
3. MIXED_ADAPTIVE: Model must detect which transform to apply
4. BELIEF_FLIP: Paper-inspired "Aha!" scenario - sudden complete reversals

Expected Results:
- GPT2:   No geometric inductive bias, learns through residual approximation
- DDL:    CAN do reflections via β→2, competitive with E∆
- Cayley: CANNOT do reflections (det=+1 constraint), FAILS on reflection tasks
- mHC:    Multi-head Householder chains, CAN do reflections
- E∆:     Adaptive gating, CAN do both rotations and reflections

============================================================================
"""

import os
import numpy as np
import argparse


def generate_rotation_matrix(dim: int, angle: float = None) -> np.ndarray:
    """Generate a proper rotation matrix (det=+1) in arbitrary dimension."""
    if angle is None:
        angle = np.random.uniform(0.1, np.pi)
    
    # Create rotation in a random 2D subspace
    Q = np.eye(dim, dtype=np.float32)
    
    # Pick two random dimensions to rotate
    i, j = np.random.choice(dim, 2, replace=False)
    
    c, s = np.cos(angle), np.sin(angle)
    Q[i, i] = c
    Q[j, j] = c
    Q[i, j] = -s
    Q[j, i] = s
    
    return Q


def generate_reflection_matrix(dim: int, v: np.ndarray = None) -> np.ndarray:
    """Generate a Householder reflection matrix (det=-1)."""
    if v is None:
        v = np.random.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    
    # Householder: H = I - 2*v*v^T
    H = np.eye(dim, dtype=np.float32) - 2 * np.outer(v, v)
    return H


def generate_pure_rotation_dataset(
    output_dir: str = 'data/correction_rotation',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    TASK 1: Pure Rotation Dataset
    
    All transformations are rotations (det=+1).
    DDL/Cayley should excel here.
    E∆ should also do well (can do rotations via Cayley branch).
    
    This establishes a baseline where rotation-based methods have advantage.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print("="*70)
    print("TASK 1: PURE ROTATION")
    print("Expected: DDL ≈ Cayley ≈ E∆ > mHC > GPT2")
    print("(All rotation-capable models should perform similarly)")
    print("="*70)
    
    n_total = n_train + n_val
    data_x, data_y = [], []
    
    for i in range(n_total):
        # Base vector
        v = np.random.randn(dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        
        # Generate rotation matrix
        angle = np.random.uniform(0.3, np.pi - 0.3)  # Avoid trivial/flip angles
        R = generate_rotation_matrix(dim, angle)
        rotated_v = R @ v
        
        # Signal position
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        # Build sequence
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # Rotation signal: negative first dimension
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = -3.0  # Rotation marker
        signal[1] = np.sin(angle)  # Encode angle info
        signal[2] = np.cos(angle)
        
        noise_level = 0.02
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < signal_pos:
                seq_x[t] = v + noise
                seq_y[t] = v
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = rotated_v  # Apply rotation
            else:
                seq_x[t] = rotated_v + noise
                seq_y[t] = rotated_v
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    X, Y = np.stack(data_x), np.stack(data_y)
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    
    print(f"Saved to {output_dir}/")
    print(f"  Train: {X[:n_train].shape}, Val: {X[n_train:].shape}")


def generate_pure_reflection_dataset(
    output_dir: str = 'data/correction_reflection',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    TASK 2: Pure Reflection Dataset (THE KEY DIFFERENTIATOR)
    
    All transformations are reflections (det=-1).
    THIS IS WHERE CAYLEY/DDL SHOULD FAIL!
    
    Mathematical fact: Cayley transform can ONLY produce rotations (det=+1).
    It CANNOT produce reflections (det=-1) - this is a topological constraint.
    
    E∆-MHC-Geo with Householder should excel here.
    This is the "Aha!" moment equivalent - instant complete belief flips.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print("="*70)
    print("TASK 2: PURE REFLECTION (Critical Differentiator!)")
    print("Expected: DDL ≈ E∆ >> mHC > GPT2 >> Cayley")
    print("Key insight: Cayley (det=+1) CANNOT do reflections (det=-1)")
    print("DDL CAN do reflections via β→2 in A = I - β·kk^T")
    print("="*70)
    
    n_total = n_train + n_val
    data_x, data_y = [], []
    
    for i in range(n_total):
        # Base vector
        v = np.random.randn(dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        
        # Generate reflection (Householder)
        # Reflect v through a random hyperplane
        reflection_axis = np.random.randn(dim).astype(np.float32)
        reflection_axis = reflection_axis / np.linalg.norm(reflection_axis)
        H = generate_reflection_matrix(dim, reflection_axis)
        reflected_v = H @ v
        
        # Verify it's actually a reflection (det=-1)
        assert abs(np.linalg.det(H) + 1) < 1e-5, "Not a reflection!"
        
        # Signal position
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        # Build sequence
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # Reflection signal: positive first dimension (distinct from rotation)
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = 5.0  # Reflection marker
        signal[1:4] = reflection_axis[:3] * 0.5  # Encode reflection axis hint
        
        noise_level = 0.02
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < signal_pos:
                seq_x[t] = v + noise
                seq_y[t] = v
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = reflected_v  # Apply reflection
            else:
                seq_x[t] = reflected_v + noise
                seq_y[t] = reflected_v
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    X, Y = np.stack(data_x), np.stack(data_y)
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    
    # Also save special negation subset: v → -v (perfect flip)
    # This is the purest "Aha!" moment test
    print("\nGenerating NEGATION subset (v → -v)...")
    negation_x, negation_y = [], []
    for i in range(n_train // 4):
        v = np.random.randn(dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = 5.0
        
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * 0.02
            if t < signal_pos:
                seq_x[t] = v + noise
                seq_y[t] = v
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = -v  # PERFECT NEGATION
            else:
                seq_x[t] = -v + noise
                seq_y[t] = -v
        
        negation_x.append(seq_x)
        negation_y.append(seq_y)
    
    np.save(os.path.join(output_dir, 'negation_x.npy'), np.stack(negation_x))
    np.save(os.path.join(output_dir, 'negation_y.npy'), np.stack(negation_y))
    
    print(f"Saved to {output_dir}/")
    print(f"  Train: {X[:n_train].shape}, Val: {X[n_train:].shape}")
    print(f"  Negation subset: {len(negation_x)} samples")


def generate_mixed_adaptive_dataset(
    output_dir: str = 'data/correction_mixed',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    TASK 3: Mixed Rotation/Reflection (Tests Adaptive Gating)
    
    Half rotations, half reflections - model must detect which to apply.
    
    E∆-MHC-Geo should excel: beta gating can adapt to signal
    - β→1: Use Cayley for rotations
    - β→0: Use Householder for reflections
    
    DDL will fail on reflection half.
    GPT2 has no geometric inductive bias.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print("="*70)
    print("TASK 3: MIXED ROTATION/REFLECTION (Tests Adaptive Gating)")
    print("Expected: DDL ≈ E∆ > mHC > GPT2 >> Cayley")
    print("Cayley fails on the reflection half")
    print("="*70)
    
    n_total = n_train + n_val
    data_x, data_y = [], []
    transform_types = []
    
    for i in range(n_total):
        v = np.random.randn(dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        
        # 50% rotation, 50% reflection
        is_reflection = np.random.random() < 0.5
        
        if is_reflection:
            # Reflection
            axis = np.random.randn(dim).astype(np.float32)
            axis = axis / np.linalg.norm(axis)
            H = generate_reflection_matrix(dim, axis)
            transformed_v = H @ v
            signal_marker = 5.0  # Positive = reflection
            transform_types.append('reflection')
        else:
            # Rotation
            angle = np.random.uniform(0.3, np.pi - 0.3)
            R = generate_rotation_matrix(dim, angle)
            transformed_v = R @ v
            signal_marker = -3.0  # Negative = rotation
            transform_types.append('rotation')
        
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = signal_marker
        
        noise_level = 0.02
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < signal_pos:
                seq_x[t] = v + noise
                seq_y[t] = v
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = transformed_v
            else:
                seq_x[t] = transformed_v + noise
                seq_y[t] = transformed_v
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    X, Y = np.stack(data_x), np.stack(data_y)
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    np.save(os.path.join(output_dir, 'transform_types.npy'), 
            np.array(transform_types[:n_train]))
    
    rot_count = transform_types[:n_train].count('rotation')
    ref_count = transform_types[:n_train].count('reflection')
    print(f"Saved to {output_dir}/")
    print(f"  Rotations: {rot_count} ({100*rot_count/n_train:.1f}%)")
    print(f"  Reflections: {ref_count} ({100*ref_count/n_train:.1f}%)")


def generate_belief_flip_dataset(
    output_dir: str = 'data/correction_belief_flip',
    dim: int = 32,
    seq_len: int = 64,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    TASK 4: Paper-Inspired "Belief Flip" / "Aha!" Moment Task
    
    Based on the Illusion of Insight paper's key finding:
    - "Wait, something is wrong" → complete reversal of reasoning
    
    This simulates the scenario where:
    1. Model builds up a "belief" (repeated exposure to concept)
    2. A correction signal indicates complete reversal needed
    3. Model must INSTANTLY flip to opposite belief
    
    The paper shows:
    - Spontaneous flips (no signal) rarely help
    - Triggered flips (explicit signal) help significantly
    - High uncertainty makes correction more valuable
    
    We test:
    - MAINTAIN: No correction signal → keep belief (test for false positives)
    - TRIGGERED: Correction signal → flip belief immediately
    - GRADUAL: Soft signal → partial correction (tests fine-grained control)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    print("="*70)
    print("TASK 4: BELIEF FLIP (Paper's 'Aha!' Moment)")
    print("Expected: E∆ >> others (instant perfect flip via Householder)")
    print("="*70)
    
    n_total = n_train + n_val
    data_x, data_y = [], []
    scenario_types = []
    
    # Scenario distribution (inspired by paper)
    # Paper: ~6.31% spontaneous shifts (we use 'maintain' to test this)
    scenarios = {
        'maintain': 0.35,      # No correction needed
        'triggered_flip': 0.45,  # Complete belief reversal
        'triggered_partial': 0.20,  # Partial correction (rotation)
    }
    
    for i in range(n_total):
        # Generate "belief" vector
        belief = np.random.randn(dim).astype(np.float32)
        belief = belief / np.linalg.norm(belief)
        
        scenario = np.random.choice(list(scenarios.keys()), p=list(scenarios.values()))
        scenario_types.append(scenario)
        
        # Variable uncertainty (paper's entropy stratification)
        uncertainty = np.random.uniform(0.1, 0.9)
        noise_level = uncertainty * 0.15
        
        # Signal position (in middle third of sequence)
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        if scenario == 'maintain':
            # No correction - model should NOT flip
            # Build up belief, maintain throughout
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                seq_x[t] = belief + noise
                seq_y[t] = belief
                
        elif scenario == 'triggered_flip':
            # THE KEY TEST: Complete belief reversal
            # This requires a REFLECTION (det=-1)
            
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = 5.0  # Strong correction signal
            
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                if t < signal_pos:
                    # Build up original belief
                    seq_x[t] = belief + noise
                    seq_y[t] = belief
                elif t == signal_pos:
                    # CORRECTION SIGNAL
                    seq_x[t] = signal
                    seq_y[t] = -belief  # COMPLETE FLIP (requires reflection!)
                else:
                    # Maintain flipped belief
                    seq_x[t] = -belief + noise
                    seq_y[t] = -belief
                    
        elif scenario == 'triggered_partial':
            # Partial correction - rotates belief by ~90 degrees
            # This CAN be done by Cayley, but Householder can also do it
            
            angle = np.random.uniform(np.pi/3, 2*np.pi/3)  # 60-120 degree rotation
            R = generate_rotation_matrix(dim, angle)
            partial_correction = R @ belief
            
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = -3.0  # Rotation signal (weaker than flip)
            signal[1] = np.sin(angle)
            
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                if t < signal_pos:
                    seq_x[t] = belief + noise
                    seq_y[t] = belief
                elif t == signal_pos:
                    seq_x[t] = signal
                    seq_y[t] = partial_correction
                else:
                    seq_x[t] = partial_correction + noise
                    seq_y[t] = partial_correction
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    X, Y = np.stack(data_x), np.stack(data_y)
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    np.save(os.path.join(output_dir, 'scenario_types.npy'), 
            np.array(scenario_types[:n_train]))
    
    # Statistics
    print(f"\nScenario distribution (train):")
    for s in scenarios:
        count = scenario_types[:n_train].count(s)
        print(f"  {s}: {count} ({100*count/n_train:.1f}%)")
    
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Architectural Correction Datasets'
    )
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'rotation', 'reflection', 'mixed', 'belief_flip'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ARCHITECTURAL CORRECTION DATASET GENERATOR")
    print("Designed to expose differences between:")
    print("  - GPT2 (no geometric bias)")
    print("  - DDL/Cayley (rotations only, det=+1)")
    print("  - mHC (approximate isometries)")  
    print("  - E∆-MHC-Geo (adaptive rotation/reflection)")
    print("="*70)
    
    if args.dataset in ['all', 'rotation']:
        print("\n")
        generate_pure_rotation_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'reflection']:
        print("\n")
        generate_pure_reflection_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'mixed']:
        print("\n")
        generate_mixed_adaptive_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'belief_flip']:
        print("\n")
        generate_belief_flip_dataset(seed=args.seed)
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print("\nExpected benchmark results:")
    print("  ROTATION task:     DDL ≈ Cayley ≈ E∆ > mHC > GPT2")
    print("  REFLECTION task:   DDL ≈ E∆ >> mHC > GPT2 >> Cayley (CANNOT reflect)")
    print("  MIXED task:        DDL ≈ E∆ > mHC > GPT2 >> Cayley")
    print("  BELIEF_FLIP task:  DDL ≈ E∆ >> others")
    print("\nKey insight: Cayley (det=+1) CANNOT do reflections (det=-1).")
    print("DDL (arXiv:2601.00417) CAN do reflections via β→2.")
    print("E∆-MHC-Geo uses explicit Householder for reflections.")
    print("="*70)
