# Geodesic-Delta vs Baseline: Experimental Results

This document summarizes early experiments comparing the **Geodesic-Delta architecture** (`proposed_model.py`) against the **standard GPT baseline** (`model.py`) on three benchmark tasks.

## Experimental Setup

### Model Configuration (Both Architectures)
| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Heads | 4 |
| Embedding Dim | 128 |
| Dropout | 0.0 |
| Bias | False |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Weight Decay | 0.1 |
| Gradient Clip | 1.0 |
| Beta1, Beta2 | 0.9, 0.99 |
| Batch Size | 64 (grok/erasure), 16 (isometry) |
| Block Size | 128 (grok/erasure), 512 (isometry) |

### Hardware
- Single NVIDIA GPU
- PyTorch 2.0+ with `torch.compile`

---

## Task 1: Grokking (Modular Arithmetic)

**Dataset:** `a + b = c (mod 97)` for all `a, b ∈ [0, 96]`
- 9,409 examples total
- 90/10 train/val split
- ASCII character encoding

**Training:** 3,000 iterations

### Loss Progression

| Step | Geodesic Train | Geodesic Val | Baseline Train | Baseline Val |
|------|----------------|--------------|----------------|--------------|
| 0 | 3.91 | 3.93 | 3.91 | 3.91 |
| 500 | 2.34 | 2.34 | **2.15** | **2.16** |
| 1000 | 2.08 | 2.09 | **1.85** | **1.86** |
| 1500 | 1.89 | 1.90 | **1.70** | **1.71** |
| 2000 | 1.73 | 1.73 | **1.63** | **1.64** |
| 2500 | 1.63 | 1.63 | **1.59** | **1.59** |
| 3000 | 1.595 | 1.597 | **1.584** | **1.585** |

### Analysis
- **Winner:** Baseline (marginally)
- **Observations:** 
  - Baseline converges faster throughout training
  - Final loss difference is minimal (~0.01)
  - Neither model exhibits the "grokking" phenomenon (sudden generalization) within 3k iterations
  - Extended training (10k+ iterations) may be needed to observe late-stage effects

---

## Task 2: Erasure ("Did I Stutter?")

**Dataset:** Command sequences with optional negation
- Standard: `"Go North. -> North"`
- Negated: `"Go North. Wait, go South. -> South"`
- 50,000 examples (50% each type)
- 90/10 train/val split

**Training:** 5,000 iterations

### Loss Progression

| Step | Geodesic Train | Geodesic Val | Baseline Train | Baseline Val |
|------|----------------|--------------|----------------|--------------|
| 0 | 10.82 | 10.82 | 10.81 | 10.81 |
| 500 | 1.24 | 1.24 | **0.96** | **0.96** |
| 1000 | 0.34 | 0.34 | **0.20** | **0.20** |
| 1500 | 0.20 | 0.20 | 0.19 | 0.19 |
| 2000 | 0.21 | 0.21 | 0.19 | 0.19 |
| 3000 | 0.19 | 0.19 | 0.19 | 0.19 |
| 4000 | 0.19 | 0.19 | 0.19 | 0.19 |
| 5000 | 0.187 | **0.186** | **0.185** | 0.186 |

### Analysis
- **Winner:** Tie (essentially identical final performance)
- **Observations:**
  - Baseline converges faster in early training (2x faster to low loss)
  - Both models plateau at ~0.186 loss
  - The erasure/negation task may be too simple for 4-layer models
  - The Geodesic thermodynamic gating doesn't provide advantage here

---

## Task 3: Isometry (Pass-Key / Needle-in-Haystack)

**Dataset:** Random noise with embedded passkey
- Format: `"[noise]The magic number is XXXX.[noise]\nWhat is the magic number? XXXX\n"`
- Context lengths: 512, 1024, 2048, 4096 tokens (curriculum)
- 2,000 examples total
- 90/10 train/val split

**Training:** 5,000 iterations, block_size=512

### Loss Progression

| Step | Geodesic Train | Geodesic Val | Baseline Train | Baseline Val |
|------|----------------|--------------|----------------|--------------|
| 0 | 10.83 | 10.83 | 10.85 | 10.85 |
| 500 | 4.59 | 4.59 | 4.59 | 4.59 |
| 1000 | 4.56 | 4.56 | **4.47** | **4.47** |
| 1500 | 4.50 | 4.50 | **4.44** | **4.44** |
| 2000 | 4.49 | 4.48 | 4.46 | 4.46 |
| 2500 | 4.48 | 4.47 | 4.45 | **4.44** |
| 3000 | 4.45 | 4.46 | **4.44** | **4.43** |
| 4000 | 4.46 | **4.45** | 4.44 | 4.45 |
| 5000 | **4.43** | **4.44** | 4.44 | 4.44 |

### Analysis
- **Winner:** Tie (virtually identical)
- **Observations:**
  - Both models struggle with this task (high final loss ~4.4)
  - The 4-layer model may lack capacity for long-range retrieval
  - Geodesic shows slightly better final train loss but difference is negligible
  - Larger models and longer training needed for meaningful comparison

---

## Summary & Conclusions

### Performance Comparison

| Task | Geodesic Final | Baseline Final | Difference | Winner |
|------|----------------|----------------|------------|--------|
| Grokking | 1.597 | 1.585 | +0.012 | Baseline |
| Erasure | 0.186 | 0.186 | ±0.000 | Tie |
| Isometry | 4.444 | 4.443 | +0.001 | Tie |

### Computational Cost
| Metric | Geodesic | Baseline |
|--------|----------|----------|
| Time/iter (grok) | ~19ms | ~7ms |
| Parameters/block | +2.36M | - |
| Memory overhead | ~15% more | - |

### Key Findings

1. **No clear advantage for Geodesic-Delta** on small-scale experiments
2. **Baseline converges faster** in early training across all tasks
3. **Final performance is essentially identical** (within noise)
4. **Geodesic overhead** (~2.7x slower) not justified by current results

### Hypotheses for Similar Performance

1. **Scale limitations:** 4-layer, 128-dim models may be too small to benefit from geodesic constraints
2. **Task simplicity:** All three tasks may be solvable without sophisticated representation geometry
3. **Training length:** Geodesic benefits may emerge in late-stage training (grokking at 10k+ iters)
4. **Sequence length:** block_size=128-512 may not stress isometry properties sufficiently

### Recommended Future Experiments

1. **Scale up models:** Test with 12+ layers, 768+ embedding dimension
2. **Extended training:** Run grokking task for 10,000+ iterations to observe late generalization
3. **Longer sequences:** Test isometry with block_size=2048+ 
4. **Real language tasks:** Evaluate on actual language modeling (OpenWebText, etc.)
5. **Ablation studies:** Isolate contribution of each Geodesic component

---

## Recommended Next Experiments

Based on the theoretical claims of the Geodesic-Delta architecture, the following experiments are designed to stress-test conditions where the method should show advantages.

### Experiment A: Extended Grokking (Late Generalization)

**Hypothesis:** Geodesic-Delta should exhibit faster/smoother "grokking" — the sudden generalization that occurs after extended training on algorithmic tasks.

```python
# config/train_grok_extended.py
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 15000        # 5x longer training
eval_interval = 500
lr_decay_iters = 15000
min_lr = 1e-5            # Lower final LR
```

**What to look for:**
- Sudden drop in validation loss after ~5000 iterations
- Geodesic should "grok" earlier than baseline
- Monitor β (thermodynamic gate) values during training

---

### Experiment B: Scaled-Up Model

**Hypothesis:** Representation drift accumulates across layers; Geodesic rotations should help more with deeper models.

```python
# config/train_grok_large.py
n_layer = 12             # 3x more layers
n_head = 8
n_embd = 512             # 4x larger embedding
batch_size = 32          # Smaller batch for memory
max_iters = 10000
learning_rate = 3e-4     # Lower LR for larger model
```

**What to look for:**
- Loss gap between Geodesic and baseline should widen with depth
- Gradient statistics (clip events, magnitudes)
- Training stability differences

---

### Experiment C: Long-Context Isometry

**Hypothesis:** Geodesic isometry properties should preserve information over longer distances.

```python
# config/train_isometry_long.py
block_size = 2048        # 4x longer context
batch_size = 4           # Smaller batch for memory
n_layer = 6
n_head = 8
n_embd = 256
max_iters = 10000
```

**What to look for:**
- Accuracy on retrieving passkeys at different depths
- Loss vs. needle insertion depth correlation
- Geodesic should maintain performance at deeper insertions

---

### Experiment D: Complex Erasure (Multi-Step Negation)

**Hypothesis:** Geodesic "pause and think" mechanism should handle complex negation chains better.

```python
# Enhanced erasure dataset with chains:
# "Go North. Wait, go South. No, go East. Correction, go West. -> West"

# Modify data/erasure/prepare.py to generate:
# - 2-4 negation steps
# - Longer command sequences
# - More confusing distractors
```

**What to look for:**
- Accuracy vs. number of negation steps
- Geodesic should degrade more gracefully with complexity

---

### Experiment E: OpenWebText Language Modeling

**Hypothesis:** Real language modeling at scale may reveal cumulative benefits.

```python
# config/train_owt_geodesic.py
dataset = 'openwebtext'
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 12
block_size = 1024
max_iters = 100000
gradient_accumulation_steps = 40
```

**What to look for:**
- Final perplexity comparison
- Training stability (loss spikes, gradient norms)
- Sample quality differences

---

### Experiment F: Ablation Studies

Isolate the contribution of each Geodesic component:

| Variant | GeodesicDelta | mHC Mixing | Thermodynamic Gate |
|---------|---------------|------------|-------------------|
| Full Geodesic | ✅ | ✅ | ✅ |
| No Gate | ✅ | ✅ | ❌ (damper=1) |
| No Mixing | ✅ | ❌ (identity) | ✅ |
| Rotation Only | ✅ | ❌ | ❌ |
| Baseline | ❌ | ❌ | ❌ |

**What to look for:**
- Which component contributes most to any observed benefits
- Interaction effects between components

---

### Experiment G: Curriculum Learning on Isometry

**Hypothesis:** Geodesic should transfer better across context lengths.

```python
# Train on short sequences, evaluate on long:
# Phase 1: block_size=256, 5000 iters
# Phase 2: block_size=512, 5000 iters  
# Phase 3: block_size=1024, 5000 iters
# Evaluate: block_size=2048 (zero-shot transfer)
```

**What to look for:**
- Generalization to unseen context lengths
- Geodesic isometry should enable better length transfer

---

### Quick Reference: Config Files to Create

```bash
# Create extended experiment configs
cp config/train_grok.py config/train_grok_extended.py
# Edit: max_iters=15000, lr_decay_iters=15000

cp config/train_grok.py config/train_grok_large.py
# Edit: n_layer=12, n_embd=512, n_head=8

cp config/train_isometry_geodesic.py config/train_isometry_long.py
# Edit: block_size=2048, batch_size=4, n_layer=6
```

---

## Reproducing Results

```bash
# Install dependencies
pip install torch numpy

# Prepare all datasets
python data/grokking/prepare.py
python data/erasure/prepare.py
python data/isometry/prepare.py

# Run all experiments
# Grokking
python train_geodesic.py config/train_grok.py
python train.py config/train_grok_baseline.py

# Erasure
python train_geodesic.py config/train_erasure_geodesic.py
python train.py config/train_erasure_baseline.py

# Isometry
python train_geodesic.py config/train_isometry_geodesic.py
python train.py config/train_isometry_baseline.py
```

Checkpoints saved to `out-{task}-{model}/ckpt.pt`.
