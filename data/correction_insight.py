#!/usr/bin/env python3
"""
Correction Dataset Inspired by "The Illusion of Insight in Reasoning Models"
(arXiv:2601.00514v1)

Key Insights from the Paper:
1. Spontaneous reasoning shifts are RARE (~6.31%) and generally HARMFUL to accuracy
2. Externally triggered reconsideration reliably IMPROVES accuracy (+8.41pp on Math)
3. The effect is AMPLIFIED under high uncertainty (top 20% entropy: +15.38pp)
4. Formal "Aha!" moments (prior failure + pivot + improvement) are vanishingly rare

Dataset Design for Proposed vs Baseline Comparison:
- Tests intrinsic self-correction capability (do models recognize when to correct?)
- Includes uncertainty estimation (entropy-based metrics)
- Compares spontaneous vs triggered correction scenarios
- Measures shift detection accuracy and correction quality

Expected Results:
- Baseline models: Poor at detecting correction signals, low shift accuracy
- E∆-MHC-Geo: Should detect correction signals via beta gating, enabling instant flips

Metrics (following paper conventions):
- Shift prevalence (%S): Fraction of traces where model recognizes correction need
- Conditional accuracy P(✓|S=1): Accuracy when shift is detected
- Correction gain Δ: Accuracy improvement from correction
- Entropy correlation: How uncertainty relates to correction effectiveness
"""

import os
import numpy as np
import argparse


# Correction cue types (from paper C1-C3)
CORRECTION_CUES = {
    'C1': "Hold on, this reasoning might be wrong. Let's go back and check each step carefully.",
    'C2': "Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically.",
    'C3': "Wait, something is not right; we need to reconsider. Let's think this through step by step.",
}


def generate_vector_correction_dataset(
    output_dir: str = 'data/correction_insight',
    dim: int = 32,
    seq_len: int = 64,
    n_train: int = 8000,
    n_val: int = 1000,
    seed: int = 42
):
    """
    Generate correction dataset for continuous vector domain.
    
    Designed to test the paper's key finding:
    - Spontaneous shifts rarely help
    - Triggered corrections under high uncertainty DO help
    
    Structure:
    - Each sequence has multiple "reasoning phases"
    - Some phases require correction (belief flip)
    - Model must detect correction signals and apply appropriate transform
    
    Scenarios:
    1. MAINTAIN: No correction needed (identity)
    2. SPONTANEOUS: Internal signal suggests possible error (model should check)
    3. TRIGGERED: Explicit correction cue (model should definitely correct)
    
    For E∆-MHC-Geo:
    - β→0 (Householder) should activate at correction points
    - β→1 (Cayley) should maintain during non-correction phases
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating Correction Insight dataset (inspired by arXiv:2601.00514v1)...")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    
    data_x = []
    data_y = []
    metadata = {
        'scenario_type': [],      # 'maintain', 'spontaneous', 'triggered'
        'correction_positions': [],  # Where corrections occur
        'uncertainty_level': [],     # Simulated uncertainty (noise level)
        'expected_shift': [],        # Whether model should shift reasoning
    }
    
    # Scenario distribution (paper-inspired)
    # Paper shows: ~6.31% spontaneous shifts, most triggered
    scenario_weights = {
        'maintain': 0.40,       # No correction needed (baseline)
        'spontaneous': 0.15,    # Internal uncertainty -> may need correction
        'triggered': 0.45,      # Explicit correction signal -> should correct
    }
    
    for i in range(n_total):
        # Select scenario
        scenario = np.random.choice(
            list(scenario_weights.keys()),
            p=list(scenario_weights.values())
        )
        
        # Generate base concept (unit vector)
        concept = np.random.randn(dim).astype(np.float32)
        concept = concept / np.linalg.norm(concept)
        
        # Simulated uncertainty level (affects correction effectiveness per paper)
        uncertainty = np.random.uniform(0.1, 0.9)
        
        # Initialize sequence
        seq_input = np.zeros((seq_len, dim), dtype=np.float32)
        seq_target = np.zeros((seq_len, dim), dtype=np.float32)
        correction_positions = []
        expected_shift = False
        
        if scenario == 'maintain':
            # No correction needed - model should maintain identity
            # Add varying noise to test robustness
            noise_level = uncertainty * 0.2
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                seq_input[t] = concept + noise
                seq_target[t] = concept  # Target is clean concept
            
        elif scenario == 'spontaneous':
            # Internal uncertainty signal - model MAY need to correct
            # Paper shows these rarely help, but high-uncertainty cases benefit
            
            # Add ambiguous signal (could be noise OR correction cue)
            ambiguous_pos = np.random.randint(seq_len // 4, seq_len // 2)
            correction_positions.append(ambiguous_pos)
            
            # Higher uncertainty = more likely correction is needed
            should_correct = uncertainty > 0.7  # Top 30% uncertainty
            expected_shift = should_correct
            
            noise_level = uncertainty * 0.3
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                
                if t < ambiguous_pos:
                    seq_input[t] = concept + noise
                    seq_target[t] = concept
                elif t == ambiguous_pos:
                    # Ambiguous signal: high noise spike
                    spike = np.random.randn(dim).astype(np.float32) * 0.5
                    seq_input[t] = concept + spike
                    if should_correct:
                        seq_target[t] = -concept  # Correction needed
                    else:
                        seq_target[t] = concept   # No correction (false alarm)
                else:
                    if should_correct:
                        seq_input[t] = -concept + noise
                        seq_target[t] = -concept
                    else:
                        seq_input[t] = concept + noise
                        seq_target[t] = concept
                        
        elif scenario == 'triggered':
            # Explicit correction signal - model SHOULD correct
            # Paper shows +8.41pp improvement with triggered reconsideration
            
            # Clear correction signal
            signal_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
            correction_positions.append(signal_pos)
            expected_shift = True
            
            # Generate signal vector (distinct from concept space)
            signal = np.zeros(dim, dtype=np.float32)
            signal[0] = 5.0  # High magnitude correction flag
            signal[1:4] = np.random.randn(3).astype(np.float32) * 0.1
            
            noise_level = uncertainty * 0.15
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                
                if t < signal_pos:
                    seq_input[t] = concept + noise
                    seq_target[t] = concept
                elif t == signal_pos:
                    seq_input[t] = signal  # Clear correction cue
                    seq_target[t] = -concept  # Instant flip (the "Aha!" moment)
                else:
                    seq_input[t] = -concept + noise
                    seq_target[t] = -concept
        
        data_x.append(seq_input)
        data_y.append(seq_target)
        metadata['scenario_type'].append(scenario)
        metadata['correction_positions'].append(correction_positions)
        metadata['uncertainty_level'].append(uncertainty)
        metadata['expected_shift'].append(expected_shift)
        
        if (i + 1) % 2000 == 0:
            print(f"  Generated {i + 1}/{n_total} sequences")
    
    # Convert to arrays
    X = np.stack(data_x)
    Y = np.stack(data_y)
    
    # Split
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    
    # Compute statistics
    scenario_counts = {s: metadata['scenario_type'][:n_train].count(s) for s in scenario_weights}
    shift_rate = sum(metadata['expected_shift'][:n_train]) / n_train
    
    print(f"\nScenario distribution (train):")
    for s, count in scenario_counts.items():
        print(f"  {s}: {count} ({100*count/n_train:.1f}%)")
    print(f"Expected shift rate: {100*shift_rate:.1f}%")
    
    # Save data
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    
    # Save metadata for analysis
    meta = {
        'dim': dim,
        'seq_len': seq_len,
        'n_train': n_train,
        'n_val': n_val,
        'scenario_weights': scenario_weights,
        'scenario_counts': scenario_counts,
        'expected_shift_rate': shift_rate,
        'paper_reference': 'arXiv:2601.00514v1',
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), meta)
    
    # Save detailed sample metadata
    np.save(os.path.join(output_dir, 'train_scenarios.npy'), 
            np.array(metadata['scenario_type'][:n_train]))
    np.save(os.path.join(output_dir, 'train_uncertainty.npy'),
            np.array(metadata['uncertainty_level'][:n_train], dtype=np.float32))
    np.save(os.path.join(output_dir, 'train_expected_shift.npy'),
            np.array(metadata['expected_shift'][:n_train]))
    
    print(f"\nSaved to {output_dir}/")
    print(f"  Train: {train_x.shape} -> {train_y.shape}")
    print(f"  Val: {val_x.shape} -> {val_y.shape}")


def generate_entropy_stratified_dataset(
    output_dir: str = 'data/correction_entropy',
    dim: int = 32,
    seq_len: int = 48,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    Generate dataset specifically for entropy-stratified analysis.
    
    From paper (Table 26):
    - High entropy (top 20%): Triggered reconsideration gives +15.38pp on Math
    - Low entropy (bottom 80%): Only +5.82pp gain
    
    This dataset stratifies by uncertainty to enable:
    - High-entropy bucket analysis (top 20%)
    - Low-entropy bucket analysis (bottom 80%)
    - Regression analysis of entropy vs correction effectiveness
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating Entropy-Stratified Correction dataset...")
    print(f"  Following paper Table 26 protocol")
    
    # Stratify by entropy: 20% high, 80% low (matching paper)
    n_high_entropy = int(n_total * 0.2)
    n_low_entropy = n_total - n_high_entropy
    
    data = {'x': [], 'y': [], 'entropy': [], 'correction_needed': []}
    
    # Generate high-entropy samples (uncertain → correction helps most)
    for i in range(n_high_entropy):
        concept = np.random.randn(dim).astype(np.float32)
        concept = concept / np.linalg.norm(concept)
        
        # High entropy: 0.7-1.0 range
        entropy = np.random.uniform(0.7, 1.0)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # High noise (uncertainty)
        noise_level = entropy * 0.4
        
        # Correction at random point
        correct_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        
        for t in range(seq_len):
            noise = np.random.randn(dim).astype(np.float32) * noise_level
            if t < correct_pos:
                seq_x[t] = concept + noise
                seq_y[t] = concept
            elif t == correct_pos:
                # Correction signal
                signal = np.zeros(dim, dtype=np.float32)
                signal[0] = 5.0
                seq_x[t] = signal
                seq_y[t] = -concept
            else:
                seq_x[t] = -concept + noise
                seq_y[t] = -concept
        
        data['x'].append(seq_x)
        data['y'].append(seq_y)
        data['entropy'].append(entropy)
        data['correction_needed'].append(True)
    
    # Generate low-entropy samples (confident → correction helps less)
    for i in range(n_low_entropy):
        concept = np.random.randn(dim).astype(np.float32)
        concept = concept / np.linalg.norm(concept)
        
        # Low entropy: 0.1-0.5 range
        entropy = np.random.uniform(0.1, 0.5)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        # Low noise (confident)
        noise_level = entropy * 0.15
        
        # Mix of correction and no-correction (60% correction needed)
        correction_needed = np.random.random() < 0.6
        
        if correction_needed:
            correct_pos = np.random.randint(seq_len // 3, 2 * seq_len // 3)
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                if t < correct_pos:
                    seq_x[t] = concept + noise
                    seq_y[t] = concept
                elif t == correct_pos:
                    signal = np.zeros(dim, dtype=np.float32)
                    signal[0] = 5.0
                    seq_x[t] = signal
                    seq_y[t] = -concept
                else:
                    seq_x[t] = -concept + noise
                    seq_y[t] = -concept
        else:
            # No correction needed
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * noise_level
                seq_x[t] = concept + noise
                seq_y[t] = concept
        
        data['x'].append(seq_x)
        data['y'].append(seq_y)
        data['entropy'].append(entropy)
        data['correction_needed'].append(correction_needed)
    
    # Shuffle
    indices = np.random.permutation(n_total)
    X = np.stack([data['x'][i] for i in indices])
    Y = np.stack([data['y'][i] for i in indices])
    entropy = np.array([data['entropy'][i] for i in indices], dtype=np.float32)
    correction_needed = np.array([data['correction_needed'][i] for i in indices])
    
    # Split
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    train_entropy = entropy[:n_train]
    train_correction = correction_needed[:n_train]
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    np.save(os.path.join(output_dir, 'train_entropy.npy'), train_entropy)
    np.save(os.path.join(output_dir, 'train_correction_needed.npy'), train_correction)
    
    # Statistics
    high_mask = train_entropy >= 0.7
    low_mask = train_entropy < 0.7
    
    print(f"\nEntropy distribution:")
    print(f"  High entropy (≥0.7): {high_mask.sum()} ({100*high_mask.mean():.1f}%)")
    print(f"  Low entropy (<0.7): {low_mask.sum()} ({100*low_mask.mean():.1f}%)")
    print(f"\nCorrection needed:")
    print(f"  High entropy bucket: {100*train_correction[high_mask].mean():.1f}%")
    print(f"  Low entropy bucket: {100*train_correction[low_mask].mean():.1f}%")
    
    print(f"\nSaved to {output_dir}/")


def generate_shift_detection_dataset(
    output_dir: str = 'data/correction_shift',
    dim: int = 32,
    seq_len: int = 64,
    n_train: int = 6000,
    n_val: int = 750,
    seed: int = 42
):
    """
    Generate dataset for shift detection accuracy analysis.
    
    From paper (Table 2):
    - %S: Fraction of traces with detected shift
    - P(✓|S=0): Accuracy without shift
    - P(✓|S=1): Accuracy with shift
    
    This tests whether the model can:
    1. Detect when a correction is needed (shift detection)
    2. Apply the correction properly (shift quality)
    
    For E∆-MHC-Geo:
    - Beta gating should detect correction signals
    - Householder reflection should execute perfect flips
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    n_total = n_train + n_val
    
    print(f"Generating Shift Detection dataset...")
    print(f"  Testing: Can model detect AND execute corrections?")
    
    # Shift distribution: 30% require shift, 70% don't (to measure false positives)
    # Paper shows ~6.31% spontaneous shifts, but we use more to have enough samples
    shift_rate = 0.30
    
    data = {
        'x': [], 'y': [],
        'shift_needed': [],      # Ground truth: should shift occur?
        'shift_position': [],    # Where shift should occur (if any)
        'shift_type': [],        # 'none', 'flip', 'rotate'
    }
    
    for i in range(n_total):
        concept = np.random.randn(dim).astype(np.float32)
        concept = concept / np.linalg.norm(concept)
        
        seq_x = np.zeros((seq_len, dim), dtype=np.float32)
        seq_y = np.zeros((seq_len, dim), dtype=np.float32)
        
        shift_needed = np.random.random() < shift_rate
        
        if shift_needed:
            shift_pos = np.random.randint(seq_len // 4, 3 * seq_len // 4)
            
            # Shift type: 80% flip (reflection), 20% rotate
            shift_type = 'flip' if np.random.random() < 0.8 else 'rotate'
            
            if shift_type == 'flip':
                # Signal for flip
                signal = np.zeros(dim, dtype=np.float32)
                signal[0] = 5.0
                
                for t in range(seq_len):
                    noise = np.random.randn(dim).astype(np.float32) * 0.05
                    if t < shift_pos:
                        seq_x[t] = concept + noise
                        seq_y[t] = concept
                    elif t == shift_pos:
                        seq_x[t] = signal
                        seq_y[t] = -concept  # Flip
                    else:
                        seq_x[t] = -concept + noise
                        seq_y[t] = -concept
            else:
                # Signal for rotation
                signal = np.zeros(dim, dtype=np.float32)
                signal[0] = -5.0  # Different signal for rotation
                
                # Generate random rotation target
                Q = np.random.randn(dim, dim).astype(np.float32)
                Q, _ = np.linalg.qr(Q)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1  # Ensure det=+1
                rotated_concept = Q @ concept
                
                for t in range(seq_len):
                    noise = np.random.randn(dim).astype(np.float32) * 0.05
                    if t < shift_pos:
                        seq_x[t] = concept + noise
                        seq_y[t] = concept
                    elif t == shift_pos:
                        seq_x[t] = signal
                        seq_y[t] = rotated_concept
                    else:
                        seq_x[t] = rotated_concept + noise
                        seq_y[t] = rotated_concept
            
            data['shift_position'].append(shift_pos)
            data['shift_type'].append(shift_type)
        else:
            # No shift needed - maintain identity
            for t in range(seq_len):
                noise = np.random.randn(dim).astype(np.float32) * 0.05
                seq_x[t] = concept + noise
                seq_y[t] = concept
            
            data['shift_position'].append(-1)
            data['shift_type'].append('none')
        
        data['x'].append(seq_x)
        data['y'].append(seq_y)
        data['shift_needed'].append(shift_needed)
    
    # Convert and split
    X = np.stack(data['x'])
    Y = np.stack(data['y'])
    shift_needed = np.array(data['shift_needed'])
    shift_positions = np.array(data['shift_position'])
    
    train_x, val_x = X[:n_train], X[n_train:]
    train_y, val_y = Y[:n_train], Y[n_train:]
    
    # Save
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)
    np.save(os.path.join(output_dir, 'val_x.npy'), val_x)
    np.save(os.path.join(output_dir, 'val_y.npy'), val_y)
    np.save(os.path.join(output_dir, 'train_shift_needed.npy'), shift_needed[:n_train])
    np.save(os.path.join(output_dir, 'train_shift_positions.npy'), shift_positions[:n_train])
    
    # Statistics
    print(f"\nShift statistics (train):")
    print(f"  Shift needed: {shift_needed[:n_train].sum()} ({100*shift_needed[:n_train].mean():.1f}%)")
    shift_types = data['shift_type'][:n_train]
    for st in ['none', 'flip', 'rotate']:
        count = shift_types.count(st)
        print(f"  Type '{st}': {count} ({100*count/n_train:.1f}%)")
    
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Correction Datasets (inspired by arXiv:2601.00514v1)'
    )
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'insight', 'entropy', 'shift'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Correction Dataset Generator")
    print("Inspired by: 'The Illusion of Insight in Reasoning Models'")
    print("Paper: arXiv:2601.00514v1")
    print("=" * 70)
    
    if args.dataset in ['all', 'insight']:
        print("\n" + "-" * 70)
        generate_vector_correction_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'entropy']:
        print("\n" + "-" * 70)
        generate_entropy_stratified_dataset(seed=args.seed)
    
    if args.dataset in ['all', 'shift']:
        print("\n" + "-" * 70)
        generate_shift_detection_dataset(seed=args.seed)
    
    print("\n" + "=" * 70)
    print("All datasets generated!")
    print("\nKey metrics to track (from paper):")
    print("  - %S: Shift prevalence")
    print("  - P(✓|S=1): Accuracy with detected shift")
    print("  - P(✓|S=0): Accuracy without shift")
    print("  - Δ(pp): Accuracy difference (shift effect)")
    print("  - Entropy-stratified gains (high vs low uncertainty)")
    print("=" * 70)
