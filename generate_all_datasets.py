"""
Generate All Benchmark Datasets

This script generates all three "Kill-Shot" benchmark datasets for the
comparative study of E∆-MHC-Geo vs baselines.

DATASET SPECIFICATIONS (from comparative study table):
┌─────────────┬────────────┬──────────┬─────────┬──────────────┐
│ Dataset     │ Samples    │ Seq Len  │ Vec Dim │ Metric       │
├─────────────┼────────────┼──────────┼─────────┼──────────────┤
│ Gyroscope   │ 10,000     │ 256      │ 16      │ MSE Loss     │
│             │ (9k/1k)    │          │         │              │
├─────────────┼────────────┼──────────┼─────────┼──────────────┤
│ Correction  │ 5,000      │ 32       │ 32      │ Cosine Sim.  │
│             │ (4.5k/0.5k)│          │         │              │
├─────────────┼────────────┼──────────┼─────────┼──────────────┤
│ Stability   │ 1,000      │ 128/10k  │ 64      │ Norm Drift   │
│             │ (900/100)  │ train/test│        │              │
└─────────────┴────────────┴──────────┴─────────┴──────────────┘

Usage:
    python generate_all_datasets.py
    python generate_all_datasets.py --seed 42 --data_dir data
"""

import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Generate all benchmark datasets')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that already exist')
    args = parser.parse_args()
    
    print("=" * 70)
    print("BENCHMARK DATASET GENERATION")
    print("=" * 70)
    print(f"Output directory: {args.data_dir}")
    print(f"Random seed: {args.seed}")
    print()
    
    total_start = time.time()
    
    # ==========================================================================
    # Dataset 1: Gyroscope
    # Specification: 10,000 trajectories (9k train, 1k val), 256 steps, 16-dim
    # ==========================================================================
    gyro_dir = os.path.join(args.data_dir, 'gyroscope')
    if args.skip_existing and os.path.exists(os.path.join(gyro_dir, 'train_x.npy')):
        print("Gyroscope dataset exists, skipping...")
    else:
        print("=" * 70)
        print("Dataset 1: GYROSCOPE (Continuous Rotation)")
        print("Target: Proves MANIFOLD PRECISION vs. DDL's linear approximation")
        print("Spec: 10,000 samples (9k/1k), seq_len=256, dim=16")
        print("=" * 70)
        
        from data.gyroscope import generate_gyroscope_data
        
        t0 = time.time()
        generate_gyroscope_data(
            output_dir=gyro_dir,
            dim=16,           # Vector dimension
            seq_len=256,      # Trajectory length
            n_train=9000,     # Training samples
            n_val=1000,       # Validation samples
            theta_min=0.1,    # Min rotation angle (radians)
            theta_max=2.5,    # Max rotation angle (DDL breaks > 0.5)
            seed=args.seed
        )
        print(f"Time: {time.time() - t0:.1f}s\n")
    
    # ==========================================================================
    # Dataset 2: Correction
    # Specification: 5,000 sequences (4.5k train, 0.5k val), 32 steps, 32-dim
    # ==========================================================================
    corr_dir = os.path.join(args.data_dir, 'correction')
    if args.skip_existing and os.path.exists(os.path.join(corr_dir, 'train_x.npy')):
        print("Correction dataset exists, skipping...")
    else:
        print("=" * 70)
        print("Dataset 2: CORRECTION PROTOCOL (Logical Negation)")
        print("Target: Proves TOPOLOGICAL COMPLETENESS vs. Cayley's blind spot")
        print("Spec: 5,000 samples (4.5k/0.5k), seq_len=32, dim=32")
        print("=" * 70)
        
        from data.correction import generate_correction_data
        
        t0 = time.time()
        generate_correction_data(
            output_dir=corr_dir,
            dim=32,           # Vector dimension
            seq_len=32,       # Sequence length
            n_train=4500,     # Training samples
            n_val=500,        # Validation samples
            seed=args.seed
        )
        print(f"Time: {time.time() - t0:.1f}s\n")
    
    # ==========================================================================
    # Dataset 3: Stability
    # Specification: 1,000 sequences (900 train, 100 val), 128/10k steps, 64-dim
    # ==========================================================================
    stab_dir = os.path.join(args.data_dir, 'stability')
    if args.skip_existing and os.path.exists(os.path.join(stab_dir, 'train_x.npy')):
        print("Stability dataset exists, skipping...")
    else:
        print("=" * 70)
        print("Dataset 3: INFINITE ECHO (Stability Test)")
        print("Target: Proves UNCONDITIONAL ISOMETRY vs. baseline norm drift")
        print("Spec: 1,000 samples (900/100), seq_len=128 (train)/10k (test), dim=64")
        print("=" * 70)
        
        from data.stability import generate_stability_data, generate_long_test_sequences
        
        t0 = time.time()
        generate_stability_data(
            output_dir=stab_dir,
            dim=64,              # Vector dimension
            n_train=900,         # Training samples
            n_val=100,           # Validation samples
            train_seq_len=128,   # Training sequence length
            noise_scale=0.001,   # Small noise for active denoising
            seed=args.seed
        )
        
        # Also generate long test sequences for stability analysis
        generate_long_test_sequences(
            output_dir=stab_dir,
            dim=64,
            n_sequences=100,
            seq_len=10000,       # Long horizon for norm drift test
            noise_scale=0.0,     # Pure stability test (no noise)
            seed=args.seed + 1000
        )
        print(f"Time: {time.time() - t0:.1f}s\n")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {time.time() - total_start:.1f}s")
    print()
    print("Generated datasets:")
    print()
    print("  1. GYROSCOPE (Rotation Prediction)")
    print(f"     Path: {gyro_dir}/")
    print("     Samples: 10,000 (9k train, 1k val)")
    print("     Dimensions: 16-dim vectors, 256 steps")
    print("     Angles: 0.1 to 2.5 radians")
    print()
    print("  2. CORRECTION (Negation/Belief Flip)")
    print(f"     Path: {corr_dir}/")
    print("     Samples: 5,000 (4.5k train, 0.5k val)")
    print("     Dimensions: 32-dim vectors, 32 steps")
    print("     Signal position: Variable (25%-75% of sequence)")
    print()
    print("  3. STABILITY (Norm Preservation)")
    print(f"     Path: {stab_dir}/")
    print("     Samples: 1,000 (900 train, 100 val)")
    print("     Dimensions: 64-dim vectors")
    print("     Training: 128 steps, Testing: 10,000 steps")
    print()
    print("=" * 70)
    print("FAIR FIGHT MODEL CONFIGURATION")
    print("=" * 70)
    print("""
All models use identical architecture parameters:
    n_layer     = 6       # Depth (enough to show drift/collapse)
    n_embd      = 128     # Width
    n_head      = 4       # Attention heads
    n_streams   = 4       # For mHC and E∆-MHC-Geo
    dropout     = 0.0     # Disabled for geometric precision
    bias        = False   # Clean signal path
    batch_size  = 64
    lr          = 1e-3    # Aggressive (tests stability)
    max_iters   = 2000
    weight_decay= 0.1
    grad_clip   = 1.0
""")
    print("=" * 70)
    print("EXPECTED RESULTS (Hypotheses)")
    print("=" * 70)
    print("""
Gyroscope (Rotation Prediction):
    - Standard GPT: Loss plateaus at ~0.1 (linear approximation error)
    - DDL: Unstable for large angles (θ > 0.5), may diverge
    - mHC: Higher loss due to spectral dampening
    - E∆-MHC-Geo: Near-zero loss across all angles

Correction (Negation):
    - Standard GPT: Struggles with x → -x
    - DDL: Can negate but unstable during training
    - mHC: Cannot flip instantly (no eigenvalue -1)
    - E∆-MHC-Geo: Instant flip via Householder gate

Stability (Norm Preservation):
    - Standard GPT: Norm drifts after ~500 steps
    - DDL: Norm drifts due to β ≠ 2
    - mHC: Spectral collapse (streams converge)
    - E∆-MHC-Geo: ||v|| = 1.0 for 10,000+ steps
""")
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
Run the comparative study:

    # Option 1: Run all experiments automatically
    chmod +x run_experiments.sh
    ./run_experiments.sh

    # Option 2: Run individual experiments
    python train_continuous.py --model_type gpt2 --dataset gyroscope --out_dir out-gyroscope-baseline
    python train_continuous.py --model_type ddl --dataset gyroscope --out_dir out-gyroscope-ddl
    python train_continuous.py --model_type mhc --dataset gyroscope --out_dir out-gyroscope-mhc
    python train_continuous.py --model_type edelta --dataset gyroscope --out_dir out-gyroscope-proposed

    # Evaluate and generate plots
    python evaluate_models.py --dataset gyroscope --plot
""")


if __name__ == '__main__':
    main()
