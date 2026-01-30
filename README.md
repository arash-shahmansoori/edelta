# E∆-MHC-Geo: Geodesic Manifold-Delta Transformer

A topologically complete transformer architecture operating on the full Orthogonal Group O(n).

![nanoGPT](assets/nanogpt.jpg)

## Overview

This repository implements the **E∆-MHC-Geo** (E-Delta-MHC-Geo) architecture, a novel transformer design that achieves:

- **Unconditional Orthogonality**: Cayley rotations guarantee `Q(x)ᵀQ(x) = I` for any input
- **Topological Completeness**: Full O(n) coverage via Householder reflections (det=-1)
- **Thermodynamic Gating**: Entropy-aware switching between rotation and reflection

## Project Structure

```
edelta/
├── src/                        # Main source code
│   ├── models/                 # Model implementations
│   │   ├── baseline_gpt.py     # Standard GPT baseline
│   │   ├── ddl.py              # Deep Delta Learning (arXiv:2601.00417)
│   │   ├── mhc.py              # DeepSeek mHC (arXiv:2512.24880)
│   │   └── edelta_hybrid.py    # E∆-MHC-Geo (proposed model)
│   ├── training/               # Training scripts
│   │   ├── train_continuous.py # Continuous physics benchmarks
│   │   └── train_language_model.py
│   ├── data/                   # Data preparation scripts
│   │   ├── gyroscope.py        # Manifold precision test
│   │   ├── correction.py       # Topological completeness test
│   │   └── stability.py        # Isometry test
│   └── visualization/          # Publication figure generation
│       └── visualize_journal.py
├── experiments/                # Experiment scripts
│   └── reflection_test.py      # Direct reflection capability test
├── data/                       # Generated datasets
│   ├── gyroscope/
│   ├── correction/
│   └── stability/
├── results/                    # Generated figures
│   ├── journal_fig1_training.png
│   ├── journal_fig2_stability.png
│   └── journal_fig3_ablation.png
├── docs/                       # Documentation
│   ├── RESEARCH_V3.md          # Full theoretical foundation
│   └── ...
└── archive/                    # Old/experimental code
```

## Installation

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Python and sync dependencies
uv python install
uv sync
```

## Quick Start

### 1. Prepare Datasets

```bash
# Generate gyroscope dataset (rotation manifold test)
uv run src/data/gyroscope.py

# Generate correction dataset (belief flip test)
uv run src/data/correction.py

# Generate stability dataset (norm preservation test)
uv run src/data/stability.py
```

### 2. Train Models

```bash
# Train E∆-MHC-Geo on gyroscope task
uv run src/training/train_continuous.py --model_type edelta --dataset gyroscope --out_dir out-edelta

# Train baseline GPT for comparison
uv run src/training/train_continuous.py --model_type gpt2 --dataset gyroscope --out_dir out-baseline

# Train DDL for comparison
uv run src/training/train_continuous.py --model_type ddl --dataset gyroscope --out_dir out-ddl

# Train mHC for comparison
uv run src/training/train_continuous.py --model_type mhc --dataset gyroscope --out_dir out-mhc
```

**Training on different datasets:**

```bash
# Correction task (belief flip / negation)
uv run src/training/train_continuous.py --model_type edelta --dataset correction --out_dir out-correction

# Stability task (norm preservation)
uv run src/training/train_continuous.py --model_type edelta --dataset stability --out_dir out-stability
```

### 3. Run Reflection Test

```bash
# Test direct reflection capability (geometric inductive bias)
uv run experiments/reflection_test.py --mode sample_efficiency

# Single test with specific parameters
uv run experiments/reflection_test.py --mode single --n_samples 100 --max_iters 1000
```

### 4. Generate Figures

```bash
# Generate publication-quality figures (saves to results/)
uv run src/visualization/visualize_journal.py
```

### 5. Sample from Language Model

```bash
# Sample from a trained language model checkpoint
uv run src/utils/sample.py --out_dir=out-shakespeare-char
```

## Key Results

See `results/` for the publication figures:

| Figure | Description |
|--------|-------------|
| `journal_fig1_training.png` | Training dynamics: loss and gradient norm evolution |
| `journal_fig2_stability.png` | Stability analysis: norm preservation test |
| `journal_fig3_ablation.png` | Final performance comparison across datasets |

## Model Comparison

| Model | Architecture | Key Property |
|-------|--------------|--------------|
| **GPT** | `x + MLP(x)` | Standard residual |
| **DDL** | `x - β(k·x)k` | Rank-1 linear update |
| **mHC** | Sinkhorn doubly stochastic | Approximate orthogonality |
| **E∆-MHC-Geo** | `γ·Cayley + (1-γ)·Householder` | Exact O(n) coverage |

## Theoretical Foundation

The E∆-MHC-Geo architecture is built on:

1. **Data-Dependent Cayley Transform** (Definition 2.3):
   - `Q(x) = (I + (β/2)A(x))⁻¹(I - (β/2)A(x))`
   - Unconditionally orthogonal for ANY β

2. **Householder Reflection** (Theorem 7):
   - `H₂(k) = I - 2·k·kᵀ`
   - β=2 is FIXED (only value achieving both orthogonality AND negation)

3. **DDC-Hybrid Operator** (Definition 5.2):
   - `G_γ(X) = γ·Q(X)·X + (1-γ)·H₂(k(X))·X`
   - Full O(n) coverage via thermodynamic gating

See `docs/RESEARCH_V3.md` for the complete mathematical foundation.

## Training Options

```bash
# Full list of training options
uv run src/training/train_continuous.py --help

# Common options:
#   --model_type    gpt2, ddl, mhc, edelta
#   --dataset       gyroscope, correction, stability
#   --out_dir       Output directory for checkpoints
#   --max_iters     Number of training iterations (default: 2000)
#   --batch_size    Batch size (default: 64)
#   --n_layer       Number of transformer layers (default: 6)
#   --n_embd        Embedding dimension (default: 128)
#   --learning_rate Learning rate (default: 1e-3)
#   --device        cuda or cpu (default: cuda)
```

## References

- DDL: arXiv:2601.00417
- DeepSeek mHC: arXiv:2512.24880
- Cayley Transform: Cayley, A. (1846)
- Householder Reflection: Householder, A.S. (1958)

## License

MIT License - see LICENSE file.
