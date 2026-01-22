# Geodesic-Delta Diagnostic Findings

## Executive Summary

After implementing comprehensive diagnostics and ablation studies, we found that the **Geodesic-Delta mechanism is mathematically sound but empirically detrimental** for the tested tasks. The model consistently learns to **close the gate** (β→0), effectively disabling the rotation mechanism. When forced open, performance degrades significantly.

---

## Diagnostic Experiments Conducted

### 1. Beta Logging Diagnostic

Added real-time logging of β values during training to monitor gate activity:

```
[BETA] Mean: 0.002475 | Max: 0.002475 | w_alpha: -0.006 | b_init: -6.0
```

**Finding**: β ≈ 0.0025 consistently across all experiments (gate is closed).

### 2. Differential Learning Rate (50x Boost)

Boosted geodesic parameter learning rate by 50x to encourage gate exploration.

**Finding**: β remained at ~0.0025. The model learned to push w_alpha **negative** (closing the gate harder).

### 3. Damper Ablation

Removed the `1 - tanh(β)` damping term to allow full gradient flow.

**Finding**: No improvement. β still stuck at ~0.0025.

### 4. Static Gate Ablation

Replaced thermodynamic gating with a simple learnable scalar.

**Finding**: Gate opened (β ≈ 0.69), but **loss increased from 1.58 to 1.97**.

### 5. Tiny Model Comparison

Reduced model to 1 layer, 32 dimensions to "starve" the baseline.

**Finding**: Tiny baseline (loss: 1.60) **outperformed** tiny geodesic (loss: 2.52).

---

## Quantitative Results

| Experiment | Beta Value | Final Loss | vs Baseline |
|------------|------------|------------|-------------|
| Baseline | N/A | **1.58** | — |
| Original Geodesic | 0.0025 | 1.60 | -1.3% |
| Boosted (50x LR) | 0.0025 | 1.75 | -10.8% |
| No Damper | 0.0025 | 2.33 | -47.5% |
| **Static Gate (forced open)** | **0.69** | **1.97** | **-24.7%** |
| Tiny Baseline | N/A | **1.60** | — |
| Tiny Geodesic | 0.0016 | 2.52 | -57.5% |

---

## 🔴 Critical Discovery: Rotation Generator Collapse

### Finding (January 22, 2026)

Deep analysis of trained checkpoints revealed that **the model doesn't just close the gate (β→0), it ZEROS the rotation generators (u, v→0)!**

```
Layer 0 Attention:
  ||u||: 0.000214    (should be ~0.01)
  ||v||: 0.000199    (should be ~0.01)
  ||A||_F: 0.000000  (rotation strength = ZERO!)

Layer 1+:
  ||u|| ≈ 0.000001   (essentially zero)
  ||v|| ≈ 0.000001
```

### Why This Matters

Even if β > 0, since A = uv^T - vu^T:
- If u = v = 0, then A = 0
- If A = 0, then Q = (I+βA)^{-1}(I-βA) = I (identity!)
- **The model found a loophole to disable rotation entirely**

### Measured Rotation Magnitude

```
||Qx - x|| = 0.000000  (exactly zero!)
```

The Cayley transform is correctly implemented, but there's **nothing to rotate** because the generators collapsed.

### Why Generators Collapse

1. **Weight Decay**: With u, v initialized at 0.01, weight decay pushes them toward 0
2. **No Gradient Signal**: If rotation doesn't help, gradients for u, v are near zero
3. **Stable Minimum**: u = v = 0 is a stable point where rotation = identity

### V2 Fix: Normalized Generators

Created `proposed_model_v2.py` with normalized u, v vectors:

```python
def get_normalized_generators(self):
    """Force unit norm to prevent collapse."""
    u = self.u_raw / (torch.norm(self.u_raw) + 1e-8)
    v = self.v_raw / (torch.norm(self.v_raw) + 1e-8)
    return u, v
```

### V2 Results: Forced Rotation is HARMFUL!

| Model | Val Loss (5000 iters) | vs Baseline |
|-------|----------------------|-------------|
| **Baseline** | **1.58** | — |
| V2 (forced rotation) | 2.32 | **+47% WORSE** |

**Critical Conclusion**: The model was collapsing u, v to zero **because that was optimal**. Rotation is actively harmful for these tasks. When we force rotation to happen, performance significantly degrades.

This proves:
1. The original model's "gate closing" behavior was **intelligent optimization**
2. The rotation geometry is **not the right inductive bias** for modular arithmetic
3. The Geodesic-Delta mechanism adds capacity that the task doesn't need and can't use

---

## Analysis: Why the Gate Closes

### The Optimization Landscape

The model faces two paths to minimize loss:

1. **Path A (Baseline)**: Use standard additive residuals. Well-understood, efficient gradient flow.
2. **Path B (Geodesic)**: Learn rotations that somehow improve representations.

**The model consistently chooses Path A** because:

1. **Identity is a stable minimum**: At β=0, the Geodesic layer is identity. This is a flat region in the loss landscape.
2. **Rotation requires coordinated learning**: The u, v generators must align to create meaningful rotations. Random rotations hurt.
3. **No gradient signal for rotation**: If the model can solve the task without rotation, there's no gradient pushing β away from 0.

### The "Gradient Shortcut" Problem

```
Standard: x → LayerNorm → Attention → Add to x
Geodesic: x → Rotate → LayerNorm → Attention → Mix → Dampen → Add to rotated x
```

The geodesic path has **more operations** between input and output. Gradients flow more easily through the simpler standard path, so the optimizer exploits this.

---

## Why Rotation May Not Suit These Tasks

### 1. Modular Arithmetic is Discrete, Rotation is Continuous

The mod 97 operation maps integers to {0, 1, ..., 96}. This is fundamentally **discrete**. The Cayley rotation operates on **continuous** embedding vectors. There's a representation mismatch.

**Analogy**: Trying to model a staircase with a smooth ramp. The ramp can approximate, but it's not the natural representation.

### 2. Circularity is in Output Space, Not Embedding Space

The "circular" structure of modular arithmetic (96 + 1 = 0) exists in the **output label space**. The transformer's embeddings don't inherently have this circular structure. A rotation in embedding space doesn't naturally correspond to modular wrap-around.

### 3. The Purity Proxy May Be Uninformative

The thermodynamic gate computes:
```
φ = 1 - ||G||²_F / (tr(G))²
```

This measures "spread" of activations across streams. But there's no theoretical reason why this should indicate when rotation helps. The model correctly learns that φ provides no useful signal (w_alpha → 0).

### 4. The Task is "Too Easy"

Both grokking and erasure tasks can be solved by standard transformers through memorization + interpolation. There's no need for geometric structure. The geodesic mechanism adds complexity without benefit.

---

## Tasks Where Rotation MIGHT Help

Unitary rotations preserve:
- **Norms** (||Qx|| = ||x||)
- **Angles** (similarity structure)
- **Reversibility** (Q⁻¹ = Qᵀ)

These properties could benefit:

### 1. Long-Range Sequence Modeling

**Hypothesis**: Over many transformer layers, representations "drift" and lose information. Orthogonal transforms prevent this drift.

**Test**: Train on tasks requiring 50+ layer reasoning (e.g., long arithmetic chains).

### 2. Compositional Reasoning with Undo/Redo

**Hypothesis**: If a task requires "undoing" a previous operation, rotation provides a natural mechanism (rotate back).

**Example tasks**:
- "Apply operation A, then B, then undo A" 
- Reversible computation simulation
- Stack-based language parsing

### 3. Tasks with Explicit Symmetry Groups

**Hypothesis**: If the data has rotational/permutation symmetry, the architecture should respect it.

**Example tasks**:
- 3D spatial reasoning (SO(3) symmetry)
- Graph neural networks (permutation symmetry)
- Music generation (transposition invariance)

### 4. Adversarial Robustness

**Hypothesis**: Orthogonal transforms preserve signal-to-noise ratio. May help resist adversarial perturbations.

### 5. Continual Learning

**Hypothesis**: Orthogonal updates may reduce catastrophic forgetting by preserving old representations while learning new ones.

---

## Proposed Architectural Modifications

### Modification A: Task-Conditional Gating

Replace thermodynamic gating with **learned task embeddings**:

```python
# Instead of purity proxy:
task_embedding = self.task_encoder(x.mean(dim=1))  # Global context
beta = F.softplus(self.gate_proj(task_embedding))
```

**Rationale**: Let the model learn WHEN to rotate based on input features, not representation statistics.

### Modification B: Rotation in Value Path Only

Apply rotation to attention VALUES, not the residual stream:

```python
# Current (problematic):
x_rotated = rotate(x)
x = x_rotated + attention(x_rotated)

# Proposed:
v_rotated = rotate(v)  # Rotate values only
attn_out = softmax(QK^T) @ v_rotated
x = x + attn_out
```

**Rationale**: Keeps main gradient path clean while allowing geometric structure in attention.

### Modification C: Learnable Rotation Basis

Instead of random u, v initialization, learn task-specific rotation bases:

```python
# Initialize from SVD of attention patterns
U, S, V = torch.svd(attention_weights)
self.rotation_basis = nn.Parameter(U[:, :n_streams])
```

**Rationale**: Align rotations with actual data structure.

### Modification D: Rotation as Skip Connection Modifier

```python
# Instead of rotating main stream:
x_standard = x + attention(layernorm(x))
x_rotated = rotate(x_standard)
x_out = x_standard + alpha * (x_rotated - x_standard)  # Interpolate
```

**Rationale**: Rotation becomes a "correction" rather than the main path.

### Modification E: Per-Head Rotation

Apply rotation only to specific attention heads:

```python
# Rotate only "geometric" heads (e.g., heads 0, 3)
for i, head in enumerate(heads):
    if i in self.geometric_heads:
        head = rotate(head)
```

**Rationale**: Specialization - some heads do geometry, others do standard attention.

### Modification F: Explicit Orthogonality Loss

Add regularization encouraging orthogonal attention matrices:

```python
# Soft orthogonality constraint
orth_loss = ||WᵀW - I||²_F
total_loss = task_loss + lambda * orth_loss
```

**Rationale**: Get benefits of orthogonality without explicit rotation layers.

---

## Recommended Next Steps

### Short-term (Validate Findings)

1. [ ] Test on actual long-range tasks (e.g., Pathfinder, Long Range Arena)
2. [ ] Test on compositional tasks with explicit undo operations
3. [ ] Implement Modification B (Value-path rotation) and compare

### Medium-term (Architecture Iteration)

4. [ ] Design task specifically suited to rotation (e.g., circular sequence prediction)
5. [ ] Implement task-conditional gating (Modification A)
6. [ ] Test orthogonality regularization (Modification F)

### Long-term (Theoretical)

7. [ ] Analyze which tasks have optimization landscapes where rotation helps
8. [ ] Develop theory connecting data symmetries to architectural inductive bias
9. [ ] Study gradient flow through rotational vs additive paths

---

## Conclusion

The Geodesic-Delta mechanism represents an interesting theoretical idea: **use geometry to structure neural network representations**. However, our experiments reveal a fundamental challenge: **the optimizer has no incentive to use the geometry when simpler paths exist**.

The mechanism is not "broken" — it correctly implements unitary rotation with thermodynamic gating. The problem is **inductive bias mismatch**: the tasks we tested don't benefit from rotation, and the optimizer correctly discovered this.

Future work should:
1. **Design tasks that require geometric structure** (don't test on tasks solvable without it)
2. **Modify the architecture to make rotation easier to use** (e.g., value-path only)
3. **Develop theoretical understanding** of when geometric inductive biases help

The findings are valuable negative results that inform future architectural design.

---

## Final Verdict: The "Smart Model" Hypothesis

### Summary of Investigation

| Investigation | Finding |
|---------------|---------|
| **Gate Activity (β)** | Stays at ~0.0025 (closed) |
| **Rotation Generators (u, v)** | Collapse to ~0 (disabled) |
| **Boosted LR (50x)** | β still closed, w_alpha goes negative |
| **Static Gate (forced β)** | Performance degrades |
| **Tiny Model** | Baseline outperforms geodesic |
| **V2 (forced rotation)** | **+47% worse than baseline** |

### The Model Actively Avoids Rotation

The model discovered multiple strategies to disable rotation:
1. **Close the gate**: β → 0 via negative b_init
2. **Zero the generators**: ||u||, ||v|| → 0 
3. **Push w_alpha negative**: Learn to close the gate harder

When we blocked these escape routes (V2 with normalized generators), performance **significantly degraded**.

### Conclusion: Rotation is Wrong for These Tasks

The Geodesic-Delta mechanism is mathematically elegant but practically useless for:
- Modular arithmetic (grokking)
- Symbolic reversibility (erasure)
- Long-context retrieval (pass-key)

The model correctly learned that rotation doesn't help and found clever ways to disable it.

### What Would Rotation Actually Help?

Based on theory, rotation might help tasks with:
- **True geometric structure** (3D spatial reasoning, image transforms)
- **Group-theoretic symmetry** (SO(3), permutation groups)
- **Norm preservation requirements** (physics simulations)

The key insight: **Don't test geometric inductive biases on non-geometric tasks.**

---

## Reversibility Task Experiment

### Task Design

We designed a task specifically intended to benefit from rotation:

1. **Path Cancellation**: `"N N E S W -> (1,1)"` - North cancels South, East cancels West
2. **Explicit Undo**: `"A B undo C -> A C"` - Undo removes previous token
3. **Rotation Composition**: `"R90 R90 R180 -> R0"` - Rotations add mod 360

**Hypothesis**: These cancellation operations should benefit from the invertible nature of unitary rotations.

### Results

| Metric | Grokking Task | Reversibility Task |
|--------|---------------|-------------------|
| **Beta (start)** | 0.00247 | 0.00247 |
| **Beta (end)** | 0.00247 (no change) | **0.00373 (+51%)** |
| **w_alpha direction** | Negative (closing) | **Positive (opening!)** |
| **Final Loss Δ** | -1.3% vs baseline | -0.9% vs baseline |

### Key Insight

**The gate DOES learn to open on reversibility tasks**, suggesting the thermodynamic signal has some validity. However, even with increased gate opening, **no performance benefit** is observed.

This suggests the problem is not the gating mechanism but the **rotation geometry itself**:
- The Cayley rotation operates on embedding vectors
- Reversibility in the task is **symbolic** (token-level undo)
- There's no direct correspondence between vector rotation and symbolic cancellation

### Implications for Future Work

The reversibility task reveals that:
1. **Thermodynamic gating works as intended** - it opens when the task has reversible structure
2. **The rotation is not the right inductive bias** - even for reversibility, embedding rotation doesn't help
3. **Need task with GEOMETRIC reversibility** - perhaps visual rotation, not symbolic undo

---

## Appendix: Code Artifacts

### Diagnostic Code Added

```python
# Global flags (proposed_model.py)
ENABLE_BETA_LOGGING = True
BETA_LOG_PROB = 0.01

# In GeodesicDelta.forward():
if ENABLE_BETA_LOGGING and self.training and random.random() < BETA_LOG_PROB:
    print(f"[BETA] Mean: {beta.mean().item():.6f} | Max: {beta.max().item():.6f} | "
          f"w_alpha: {self.w_alpha.item():.6f} | b_init: {self.b_init.item():.6f}")
```

### Ablation Configs Created

| Config | Key Settings |
|--------|--------------|
| `train_grok_boosted.py` | geo_lr_mult=50.0 |
| `train_grok_no_damper.py` | use_damper=False |
| `train_grok_static_gate.py` | use_static_gate=True |
| `train_grok_tiny_*.py` | n_layer=1, n_embd=32 |

### Key Model Modifications

1. **Differential LR**: `configure_optimizers()` now separates geodesic params
2. **Damper Ablation**: `use_damper` flag in GPTConfig
3. **Static Gate**: `use_static_gate` flag bypasses purity proxy
