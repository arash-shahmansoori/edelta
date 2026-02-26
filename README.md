# E∆-MHC-Geo: Geodesic Manifold-Delta Transformer

A topologically complete transformer architecture operating on the full Orthogonal Group O(n), featuring input-adaptive, unconditionally orthogonal residual connections via the Data-Dependent Cayley transform.

## Key Results (mean ± std, 3 seeds, ~1.79M params each)

### Main Performance

| Dataset | GPT (9L) | DDL (8L) | mHC (9L) | JPmHC (7L) | **E∆ (6L)** |
|---------|----------|----------|----------|------------|-------------|
| **Gyroscope** | 3.78±0.04e-3 | 3.36±0.13e-3 | 4.02±0.03e-3 | **3.08±0.02e-4** | 1.02±0.26e-3 |
| **Stability** | 16.4±0.7e-6 | 15.3±1.8e-6 | 8650±140e-6 | 8.19±3.69e-6 | **4.35±0.27e-6** |

E∆ achieves **3.7x improvement** over GPT on gyroscope with **33% fewer layers** and **4x narrower representation**. JPmHC (n_embd=512) leads on gyroscope due to wider per-stream expressivity; E∆ leads on stability via exact orthogonality (**1.9x** over JPmHC).

### Parameter Configuration

| Model | Layers | n_embd | d_stream | Params | Reference |
|-------|--------|--------|----------|--------|-----------|
| GPT | 9 | 128 | — | 1.780M | Baseline |
| DDL | 8 | 128 | — | 1.784M | [arXiv:2406.17550](https://arxiv.org/abs/2406.17550) |
| mHC | 9 | 128 | 32 | 1.838M | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| JPmHC | 7 | 512 | 128 | 1.771M | [arXiv:2602.18308](https://arxiv.org/abs/2602.18308) |
| **E∆** | **6** | 128 | 32 | **1.788M** | This work |

### Reflection (Negation y = -x)

| Model | 500 samples | Capability |
|-------|------------|------------|
| DDL | β→1.995, acc=0.96 | Householder only |
| JPmHC | acc=-0.30 | **Fails** (SO(n)-only) |
| **E∆** | γ→0.051, **acc=0.96** | Full O(n) via gate |

### Near-π Rotation

| Dataset | GPT | DDL | mHC | JPmHC | **E∆** |
|---------|-----|-----|-----|-------|--------|
| Single (177.6°) | 1.19e-5 | 1.11e-5 | 9.64e-3 | 6.54e-6 | **1.44e-6** |
| Multi (179.9°) | 3.34e-6 | 3.75e-6 | 1.63e-3 | 1.69e-6 | **1.24e-6** |

### Figures

| Figure | Description |
|--------|-------------|
| `journal_fig1_training.png` | Training dynamics: loss and gradient norms |
| `journal_fig2_stability.png` | Norm preservation over 100 timesteps |
| `journal_fig3_ablation.png` | Final performance comparison |
| `near_pi_rotation_comparison.png` | Near-π training + per-layer gate evolution |
| `reflection_comprehensive.png` | DDL/JPmHC/E∆ negation dynamics |
| `regularization_analysis.png` | Zero-gradient theorem visualization |

---

## Project Structure

```
edelta/
├── src/
│   ├── models/
│   │   ├── baseline_gpt.py        # GPT baseline
│   │   ├── ddl.py                 # Deep Delta Learning
│   │   ├── mhc.py                 # DeepSeek mHC (Sinkhorn)
│   │   ├── jpmhc.py               # JPmHC (iterative Cayley, faithful to paper)
│   │   ├── edelta_hybrid.py       # E∆-MHC-Geo (proposed)
│   │   └── edelta_stream.py       # E∆-Stream (per-stream F variant)
│   ├── training/
│   │   ├── train_continuous.py    # Gyroscope, stability, near-π benchmarks
│   │   ├── train_reflection.py    # Reflection/negation experiments
│   │   └── train_language_model.py
│   ├── data/
│   │   ├── gyroscope.py           # Rotation prediction dataset
│   │   ├── stability.py           # Long-horizon isometry dataset
│   │   ├── near_pi_rotation.py    # Near-π rotation datasets
│   │   └── reflection.py          # Negation task
│   ├── utils/
│   │   └── param_counter.py       # Parameter counting and matching
│   └── visualization/
│       ├── visualize_journal.py   # Main figures (Fig 1-3)
│       └── visualize_near_pi.py   # Near-π figure with gate evolution
├── scripts/
│   ├── prepare_data.sh            # Generate all datasets
│   ├── run_matched_params.sh      # All 5 models, matched params
│   ├── run_near_pi.sh             # Near-π + init robustness
│   └── run_reflection.sh          # Reflection experiments
├── paper/
│   ├── main.tex                   # Paper (TMLR format)
│   ├── main.pdf                   # Compiled PDF
│   └── references.bib             # Bibliography
├── docs/
│   ├── RESEARCH.md                # Full theoretical framework
│   └── JPMHC_COMPARISON.md        # Detailed JPmHC comparison
├── data/                          # Generated datasets
├── results/                       # Generated figures
└── pyproject.toml                 # Dependencies (managed by uv)
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/arash-shahmansoori/edelta.git
cd edelta
uv python install
uv sync
```

## Quick Reproduction

```bash
# 1. Prepare all datasets
bash scripts/prepare_data.sh

# 2. Run all experiments (single seed)
bash scripts/run_matched_params.sh

# 3. Run with 3 seeds (for mean ± std)
bash scripts/run_matched_params.sh --seeds 3

# 4. Run near-π + init robustness
bash scripts/run_near_pi.sh --seeds 3

# 5. Run reflection experiments
bash scripts/run_reflection.sh

# 6. Generate figures
uv run src/visualization/visualize_journal.py
uv run src/visualization/visualize_near_pi.py
```

## Individual Model Training

```bash
# E∆-MHC-Geo (proposed, 6 layers)
uv run src/training/train_continuous.py \
    --model_type edelta --dataset gyroscope \
    --out_dir out-matched/gyroscope-proposed

# Baselines with auto-matched parameters
uv run src/training/train_continuous.py \
    --model_type gpt2 --dataset gyroscope \
    --out_dir out-matched/gyroscope-gpt --match_proposed_params

uv run src/training/train_continuous.py \
    --model_type jpmhc --dataset gyroscope \
    --out_dir out-matched/gyroscope-jpmhc --match_proposed_params

# E∆-Stream variant (per-stream F + full O(n) geometric operator)
uv run src/training/train_continuous.py \
    --model_type edelta_stream --dataset gyroscope \
    --n_embd 512 --n_layer 6 --geo_hidden_dim 48 \
    --learning_rate 5e-4 --out_dir out-matched/gyroscope-edelta_stream
```

## Parameter Counting

```bash
# Compare all models at default config
uv run src/utils/param_counter.py --quiet

# Find matching configs for baselines
uv run src/utils/param_counter.py --find_match --quiet

# Component breakdown
uv run src/utils/param_counter.py --breakdown edelta --quiet
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_type` | `gpt2` | `gpt2`, `ddl`, `mhc`, `jpmhc`, `edelta`, `edelta_stream` |
| `--dataset` | `gyroscope` | `gyroscope`, `stability`, `near_pi_rotation`, `near_pi_rotation_multiplane` |
| `--n_embd` | `128` | Embedding dimension |
| `--n_layer` | `6` | Transformer layers |
| `--n_streams` | `4` | Streams for multi-stream models |
| `--learning_rate` | `1e-3` | Peak learning rate (cosine decay) |
| `--max_iters` | `2000` | Training iterations |
| `--seed` | `42` | Random seed |
| `--match_proposed_params` | off | Auto-scale baselines to match E∆ params |
| `--gate_reg_weight` | `0.1` | Midpoint collapse regularization (E∆) |
| `--init_gate_bias` | `0.0` | Gate initialization bias (E∆) |
| `--geo_hidden_ratio` | `4` | Geometric operator hidden dim ratio (E∆) |
| `--geo_hidden_dim` | `48` | Fused geo hidden dim (E∆-Stream) |

## Model Architectures

| Model | Residual Connection | Orthogonality | Negation (λ=-1) |
|-------|-------------------|---------------|-----------------|
| GPT | x + F(x) | None | No |
| DDL | (I - β·kk^T)·x | Only at β=2 | At β=2 |
| mHC | Sinkhorn·x + F(x) | Approximate | No |
| JPmHC | Cayley(x)·x + F(x) | Approximate (iterative) | No (SO(n) only) |
| **E∆** | **γ·Cayley(x)·x + (1-γ)·H₂(x)·x** | **Exact** | **Yes** (via gate) |
| E∆-Stream | Same operator, per-stream F | Exact | Yes |

### E∆-Stream

A per-stream compute variant that combines E∆'s full O(n) geometric operator with JPmHC-style per-stream attention/MLP efficiency. Uses stream-level Cayley rotation + full-dimensional Householder reflection + dynamic routing (H_pre/H_post). Available at `src/models/edelta_stream.py`.

## References

- **E∆-MHC-Geo**: This work
- **DDL**: [arXiv:2406.17550](https://arxiv.org/abs/2406.17550)
- **mHC**: [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- **JPmHC**: [arXiv:2602.18308](https://arxiv.org/abs/2602.18308) — Sengupta, Wang & Brunswic (2026)
- **"Illusion of Insight"**: [arXiv:2601.00514](https://arxiv.org/abs/2601.00514)

## License

MIT License — see LICENSE file.
