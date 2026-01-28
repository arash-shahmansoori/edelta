# Dataset Improvements for Meaningful Experimental Results

## Current Issues

### 1. Gyroscope (Rotation)
**Current**: Works well, E∆-MHC-Geo shows 16.8x improvement
**Issue**: Could be stronger with harder cases

### 2. Correction (Negation)  
**Current**: All models achieve ~0.999 accuracy
**Issue**: Task is too easy - models memorize the simple signal pattern

### 3. Stability (Isometry)
**Current**: Confusing results - GPT/DDL appear best
**Issue**: Test methodology doesn't match training; measures trivial identity mapping

---

## Improved Dataset Designs

### Dataset 1: Enhanced Gyroscope (Compound Rotations)

**Goal**: Test if models can handle multi-plane simultaneous rotations (harder than single-plane)

```python
# Current: Single 2D rotation plane
R = rotation_in_plane(i, j, theta)

# Improved: Compound rotation across multiple planes
R = R1 @ R2 @ R3  # Compose 3 different rotations
# Or: Random SO(n) matrix via QR decomposition
```

**New specifications**:
- **Sequence length**: 1000 steps (currently 256)
- **Rotation type**: Compound multi-plane rotations
- **Angular velocity**: Variable within sequence (accelerating/decelerating)
- **Dimension**: 32 (currently 16)
- **Key metric**: MSE at step 500, step 1000 (test accumulation)

**Why this helps**: 
- Longer sequences expose accumulation of approximation errors
- Multi-plane rotations are harder to approximate linearly
- Variable angular velocity prevents memorization

---

### Dataset 2: Hard Correction (Contextual Negation)

**Goal**: Force models to truly learn negation vs memorizing patterns

**Current problem**: Signal is too obvious (first dim = 5.0)

**Improvements**:

```python
# Improvement 1: Subtle signals (not obviously different)
signal = concept + noise  # Signal is similar to concept but with small perturbation
signal = signal / ||signal||  # Still unit norm

# Improvement 2: Delayed flip (flip doesn't happen immediately)
[concept] * 10 -> [SIGNAL] -> [concept] * 3 -> [FLIP TO -concept]
# Model must remember signal happened and apply flip later

# Improvement 3: Conditional flip (flip only if certain condition)
[concept A] -> [SIGNAL] -> depends on A's properties: flip or not

# Improvement 4: Partial negation (flip only some dimensions)
target = concept.copy()
target[0:dim//2] = -target[0:dim//2]  # Flip first half only
```

**New specifications**:
- **Sequence length**: 128 (currently 32)
- **Signal type**: Subtle perturbation, not obvious flag
- **Flip timing**: Delayed by 3-10 steps after signal
- **Flip type**: Partial negation (50% of dimensions)
- **Dimension**: 64 (currently 32)
- **Key metric**: Cosine similarity at flip point, delay handling accuracy

---

### Dataset 3: True Isometry Test (Sequence-Based)

**Goal**: Test norm preservation DURING SEQUENCE PROCESSING, not single-vector autoregression

**Current problem**: Single-vector autoregression doesn't match training

**New approach**:

```python
# Instead of autoregressive single vectors, 
# test on SEQUENCES and measure norm along the sequence

# Training data:
input_seq = [v1, v2, v3, ..., vT]  # All unit vectors
target_seq = [v1, v2, v3, ..., vT]  # Identity mapping, preserve norms

# Key metric: ||output[t]|| for each position t
# Should be 1.0 across all positions

# Harder version: Long-range identity
input_seq = [key, noise, noise, noise, ..., noise, query]
target = key if query matches else noise
# Tests if model preserves key over very long context
```

**New specifications**:
- **Task**: Long-range key retrieval (preserve info over distance)
- **Sequence length**: 512-2048 steps
- **Key position**: Random (step 1-100)
- **Query position**: End of sequence
- **Dimension**: 64
- **Noise**: Unit vectors orthogonal to key
- **Key metric**: 
  - MSE of key retrieval
  - Norm of retrieved vector
  - Cosine similarity to original key

---

### Dataset 4: NEW - Manifold Interpolation

**Goal**: Test if models can interpolate ON the manifold (not through it)

**Task**: Given points A, B on a sphere, predict the geodesic midpoint

```python
# Linear interpolation (wrong - goes through sphere interior):
midpoint_linear = (A + B) / 2
midpoint_linear = midpoint_linear / ||midpoint_linear||  # Project back

# Geodesic interpolation (correct - stays on sphere):
# Midpoint on great circle between A and B
angle = arccos(A · B)
midpoint_geodesic = (A * sin(angle/2) + B * sin(angle/2)) / sin(angle)
```

**Specifications**:
- **Input**: [A, B] (two unit vectors)
- **Target**: Geodesic midpoint on sphere
- **Dimension**: 16, 32, 64
- **Key metric**: Distance to correct geodesic midpoint

**Why this differentiates models**:
- Linear models will compute linear interpolation (wrong)
- Cayley-based models might learn the correct geodesic

---

### Dataset 5: NEW - Lie Group Operations

**Goal**: Test if models can learn group operations (composition, inverse)

**Task**: Given rotation A and B, predict A @ B (composition)

```python
# Input: [R_A (flattened), R_B (flattened)]
# Target: R_A @ R_B (flattened)

# Harder: predict inverse
# Input: [R]
# Target: R^{-1} = R^T
```

**Specifications**:
- **Rotation dimension**: 3x3 (9 parameters), 4x4 (16 parameters)
- **Input**: Two rotation matrices (flattened)
- **Target**: Product or inverse
- **Key metric**: Frobenius norm error, orthogonality of output

---

## Summary of Recommended Changes

| Dataset | Current Issue | Proposed Fix | Expected Outcome |
|---------|--------------|--------------|------------------|
| Gyroscope | Already good | Longer sequences, compound rotations | Stronger differentiation |
| Correction | Too easy | Subtle signals, delayed flip | Expose negation capability |
| Stability | Wrong methodology | Sequence-based norm tracking | Measure true isometry |
| NEW: Interpolation | - | Geodesic midpoint prediction | Test manifold awareness |
| NEW: Lie Group | - | Rotation composition | Test group structure learning |

---

## Implementation Priority

1. **Fix Stability test** (High priority - current results are misleading)
2. **Harden Correction task** (High priority - current task doesn't differentiate)
3. **Add Interpolation dataset** (Medium - new theoretical insight)
4. **Enhance Gyroscope** (Low - already shows good results)
5. **Add Lie Group dataset** (Low - advanced, for future work)
