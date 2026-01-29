#!/usr/bin/env python3
"""
"Aha!" Moment Correction Dataset

Based on "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1)

============================================================================
KEY PAPER INSIGHTS:
============================================================================

1. "Aha!" moments are defined as (Definition 1):
   - Prior failure: Model was wrong before the shift
   - Prior stability: No recent changes in approach
   - Performance gain: Shift leads to correct answer

2. Spontaneous shifts are RARE (~6.31%) and generally HARMFUL:
   - P(✓|S=1) = 2.57% (accuracy WITH shift)
   - P(✓|S=0) = 16.44% (accuracy WITHOUT shift)

3. TRIGGERED reconsideration HELPS (+8.41pp on Math):
   - Explicit cues like "Wait, let me reconsider" improve accuracy
   - High entropy (uncertainty) amplifies effect: +15.38pp for top 20%

4. This maps to GEOMETRIC REFLECTION:
   - "Wait, that's completely wrong" = instant belief FLIP = REFLECTION (det=-1)
   - Gradual refinement = ROTATION (det=+1)

============================================================================
MODEL COMPARISON:
============================================================================

| Model    | Mechanism                              | Can Reflect? |
|----------|----------------------------------------|--------------|
| GPT2     | Standard residual x + f(x)             | No inductive bias |
| DDL      | A = I - β·kk^T (rank-1 perturbation)   | YES (β→2)    |
| mHC      | Doubly stochastic mixing + Sinkhorn    | Limited      |
| Proposed | Householder H = I - 2kk^T + Cayley     | YES (exact)  |

============================================================================
DATASET DESIGN:
============================================================================

Three scenarios testing "Aha!" moment capability:

1. MAINTAIN (40%): 
   - No correction needed
   - Model should preserve belief through noise
   - Tests: False positive rate (shouldn't flip when not needed)

2. TRIGGERED_FLIP (45%):
   - Explicit correction signal → complete belief reversal
   - This is the key "Aha!" moment test
   - Model must INSTANTLY flip v → -v (requires reflection)

3. TRIGGERED_PARTIAL (15%):
   - Partial correction signal → moderate adjustment
   - Rotation suffices, reflection not required
   - Tests: Can model distinguish flip vs partial correction?

============================================================================
"""

import os
import numpy as np
import argparse


def generate_aha_moment_dataset(
    output_dir: str = 'data/correction_aha',
    dim: int = 32,
    seq_len: int = 64,
    n_train: int = 8000,
    n_val: int = 1000,
    seed: int = 42
):
    """
    Generate "Aha!" Moment dataset for testing correction capability.
    
    This tests the paper's key finding: triggered corrections help,
    especially under high uncertainty.
    
    For geometric models:
    - Complete flip (v → -v) requires REFLECTION (det=-1)
    - DDL can do this via β→2 in A = I - β·kk^T
    - Proposed (E∆) can do this via Householder H = I - 2kk^T
    - mHC uses doubly stochastic mixing (limited reflection capability)
    - GPT2 has no geometric inductive bias
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print("="*70)
    print("GENERATING 'AHA!' MOMENT DATASET")
    print("Based on: 'The Illusion of Insight' (arXiv:2601.00514v1)")
    print("="*70)
    
    # Scenario distribution
    scenarios = {
        'maintain': 0.40,          # No flip needed (test false positives)
        'triggered_flip': 0.45,    # Complete reversal (THE KEY TEST)
        'triggered_partial': 0.15, # Partial correction (rotation)
    }
    
    data_x, data_y = [], []
    scenario_types = []
    flip_positions = []
    uncertainties = []
    
    for i in range(n_total):
        # Generate belief vector (unit norm)
        belief = np.random.randn(dim).astype(np.float32)
        belief = belief / np.linalg.norm(belief)
        
        # Sample scenario
        scenario = np.random.choice(list(scenarios.keys()), p=list(scenarios.values()))
        scenario_types.append(scenario)
        
        # Uncertainty level (paper: high entropy = more benefit from correction)
        uncertainty = np.random.uniform(0.1, 0.9)
        uncertainties.append(uncertainty)
        noise_level = uncertainty * 0.15
        
        # Build sequence
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # Signal position (in middle portion of sequence)
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        flip_positions.append(signal_pos if scenario != 'maintain' else -1)
        
        if scenario == 'maintain':
            # NO CORRECTION NEEDED
            # Model should maintain belief despite noise
            # Tests: Does model incorrectly flip? (false positive)
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                seq_x[t] = belief + noise
                seq_y[t] = belief
                
        elif scenario == 'triggered_flip':
            # THE KEY "AHA!" MOMENT TEST
            # Complete belief reversal: v → -v
            # This REQUIRES reflection (det=-1)
            
            # Correction signal: high magnitude marker
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = 5.0  # Strong positive = FLIP signal
            
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                
                if t < signal_pos:
                    # Build up original belief
                    seq_x[t] = belief + noise
                    seq_y[t] = belief
                elif t == signal_pos:
                    # CORRECTION SIGNAL: "Wait, that's completely wrong!"
                    seq_x[t] = signal
                    # Target: INSTANT COMPLETE FLIP
                    seq_y[t] = -belief
                else:
                    # Maintain flipped belief
                    seq_x[t] = -belief + noise
                    seq_y[t] = -belief
                    
        elif scenario == 'triggered_partial':
            # PARTIAL CORRECTION (rotation, not full flip)
            # Tests: Can model distinguish between full flip and partial?
            
            # Generate rotation angle (60-120 degrees, not full flip)
            angle = np.random.uniform(np.pi/3, 2*np.pi/3)
            
            # Create rotated belief (2D rotation in random subspace)
            i, j = np.random.choice(dim, 2, replace=False)
            rotated_belief = belief.copy()
            c, s = np.cos(angle), np.sin(angle)
            rotated_belief[i] = c * belief[i] - s * belief[j]
            rotated_belief[j] = s * belief[i] + c * belief[j]
            rotated_belief = rotated_belief / np.linalg.norm(rotated_belief)
            
            # Rotation signal: negative magnitude marker
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = -3.0  # Negative = ROTATE signal (not flip)
            
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                
                if t < signal_pos:
                    seq_x[t] = belief + noise
                    seq_y[t] = belief
                elif t == signal_pos:
                    seq_x[t] = signal
                    seq_y[t] = rotated_belief  # Partial correction
                else:
                    seq_x[t] = rotated_belief + noise
                    seq_y[t] = rotated_belief
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    # Convert to arrays
    X = np.stack(data_x)
    Y = np.stack(data_y)
    
    # Split
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    
    # Statistics
    train_scenarios = scenario_types[:n_train]
    print(f"\nDataset Statistics:")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    
    print(f"\nScenario Distribution (train):")
    for s in scenarios:
        count = train_scenarios.count(s)
        print(f"  {s:<20}: {count:>5} ({100*count/n_train:.1f}%)")
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    
    # Save metadata for analysis
    np.save(os.path.join(output_dir, 'train_scenarios.npy'), np.array(train_scenarios))
    np.save(os.path.join(output_dir, 'train_flip_positions.npy'), np.array(flip_positions[:n_train]))
    np.save(os.path.join(output_dir, 'train_uncertainties.npy'), np.array(uncertainties[:n_train], dtype=np.float32))
    
    # Save val metadata too
    np.save(os.path.join(output_dir, 'val_scenarios.npy'), np.array(scenario_types[n_train:]))
    np.save(os.path.join(output_dir, 'val_flip_positions.npy'), np.array(flip_positions[n_train:]))
    
    print(f"\nSaved to: {output_dir}/")
    print(f"  train_x.npy: {train_x.shape}")
    print(f"  train_y.npy: {train_y.shape}")
    print(f"  val_x.npy: {val_x.shape}")
    print(f"  val_y.npy: {val_y.shape}")
    
    return output_dir


def generate_pure_flip_dataset(
    output_dir: str = 'data/correction_pure_flip',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    PURE FLIP dataset: 100% complete belief reversals.
    
    This is the most direct test of reflection capability.
    Every sample requires v → -v transformation.
    
    Expected Results:
    - DDL: Should succeed (β→2 gives det=-1)
    - Proposed (E∆): Should succeed (Householder gives det=-1)
    - mHC: Limited (doubly stochastic, not orthogonal with det=-1)
    - GPT2: No geometric inductive bias, learns via residual
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print("="*70)
    print("GENERATING PURE FLIP DATASET")
    print("100% complete belief reversals (v → -v)")
    print("="*70)
    
    data_x, data_y = [], []
    
    for i in range(n_total):
        # Generate belief vector
        belief = np.random.randn(dim).astype(np.float32)
        belief = belief / np.linalg.norm(belief)
        
        # Varying uncertainty
        uncertainty = np.random.uniform(0.1, 0.9)
        noise_level = uncertainty * 0.1
        
        # Signal position
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        # Build sequence
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # Flip signal
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = 5.0
        
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            
            if t < signal_pos:
                seq_x[t] = belief + noise
                seq_y[t] = belief
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = -belief  # COMPLETE FLIP
            else:
                seq_x[t] = -belief + noise
                seq_y[t] = -belief
        
        data_x.append(seq_x)
        data_y.append(seq_y)
    
    X, Y = np.stack(data_x), np.stack(data_y)
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    
    print(f"\nSaved to: {output_dir}/")
    print(f"  Train: {X[:n_train].shape}")
    print(f"  Val: {X[n_train:].shape}")
    
    return output_dir


def generate_entropy_stratified_flip(
    output_dir: str = 'data/correction_entropy_flip',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    Entropy-stratified flip dataset.
    
    From paper Table 26:
    - High entropy (top 20%): +15.38pp gain from triggered reconsideration
    - Low entropy (bottom 80%): +5.82pp gain
    
    This dataset stratifies by uncertainty to test if models
    show differential performance based on "confidence".
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    n_high = int(n_total * 0.2)
    n_low = n_total - n_high
    
    print("="*70)
    print("GENERATING ENTROPY-STRATIFIED FLIP DATASET")
    print("Following paper Table 26 protocol")
    print("="*70)
    
    data = {'x': [], 'y': [], 'entropy': [], 'bucket': []}
    
    # HIGH ENTROPY (uncertain) - 20%
    for i in range(n_high):
        belief = np.random.randn(dim).astype(np.float32)
        belief = belief / np.linalg.norm(belief)
        
        entropy = np.random.uniform(0.7, 1.0)  # High
        noise_level = entropy * 0.25  # More noise
        
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = 5.0
        
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < signal_pos:
                seq_x[t] = belief + noise
                seq_y[t] = belief
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = -belief
            else:
                seq_x[t] = -belief + noise
                seq_y[t] = -belief
        
        data['x'].append(seq_x)
        data['y'].append(seq_y)
        data['entropy'].append(entropy)
        data['bucket'].append('high')
    
    # LOW ENTROPY (confident) - 80%
    for i in range(n_low):
        belief = np.random.randn(dim).astype(np.float32)
        belief = belief / np.linalg.norm(belief)
        
        entropy = np.random.uniform(0.1, 0.5)  # Low
        noise_level = entropy * 0.1  # Less noise
        
        signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        signal = np.zeros(dim, dtype=np.float32)
        signal[0] = 5.0
        
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < signal_pos:
                seq_x[t] = belief + noise
                seq_y[t] = belief
            elif t == signal_pos:
                seq_x[t] = signal
                seq_y[t] = -belief
            else:
                seq_x[t] = -belief + noise
                seq_y[t] = -belief
        
        data['x'].append(seq_x)
        data['y'].append(seq_y)
        data['entropy'].append(entropy)
        data['bucket'].append('low')
    
    # Shuffle
    indices = np.random.permutation(n_total)
    X = np.stack([data['x'][i] for i in indices])
    Y = np.stack([data['y'][i] for i in indices])
    entropy = np.array([data['entropy'][i] for i in indices], dtype=np.float32)
    bucket = np.array([data['bucket'][i] for i in indices])
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), X[:n_train])
    np.save(os.path.join(output_dir, 'train_y.npy'), Y[:n_train])
    np.save(os.path.join(output_dir, 'val_x.npy'), X[n_train:])
    np.save(os.path.join(output_dir, 'val_y.npy'), Y[n_train:])
    np.save(os.path.join(output_dir, 'train_entropy.npy'), entropy[:n_train])
    np.save(os.path.join(output_dir, 'train_bucket.npy'), bucket[:n_train])
    np.save(os.path.join(output_dir, 'val_entropy.npy'), entropy[n_train:])
    
    # Stats
    high_count = (bucket[:n_train] == 'high').sum()
    low_count = (bucket[:n_train] == 'low').sum()
    print(f"\nBucket Distribution (train):")
    print(f"  High entropy: {high_count} ({100*high_count/n_train:.1f}%)")
    print(f"  Low entropy:  {low_count} ({100*low_count/n_train:.1f}%)")
    
    print(f"\nSaved to: {output_dir}/")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate "Aha!" Moment Correction Datasets'
    )
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'aha', 'pure_flip', 'entropy'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("'AHA!' MOMENT DATASET GENERATOR")
    print("Based on: 'The Illusion of Insight' (arXiv:2601.00514v1)")
    print("="*70)
    
    if args.dataset in ['all', 'aha']:
        print("\n")
        generate_aha_moment_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'pure_flip']:
        print("\n")
        generate_pure_flip_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'entropy']:
        print("\n")
        generate_entropy_stratified_flip(seed=args.seed)
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print("\nExpected Results on PURE_FLIP:")
    print("  DDL:      High flip accuracy (β→2 enables det=-1)")
    print("  Proposed: High flip accuracy (Householder enables det=-1)")
    print("  mHC:      Limited (doubly stochastic, not pure reflection)")
    print("  GPT2:     No geometric bias, learns via residual approximation")
    print("="*70)
