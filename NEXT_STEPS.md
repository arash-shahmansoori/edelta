# Geodesic-Delta Research: Next Steps & Future Directions

**Last Updated:** January 22, 2026
**Status:** Investigation Phase Complete

---

## Executive Summary

After comprehensive investigation, we found that the Geodesic-Delta mechanism:
1. ✅ Is **mathematically correct** (Cayley transform, purity proxy, etc.)
2. ❌ Is **not beneficial** for tested tasks (grokking, erasure, reversibility)
3. ❌ Model **actively disables rotation** (u,v→0, β→0)
4. ❌ **Forced rotation degrades performance** (+47% worse)

**Key Insight**: The mechanism is not "broken" — the **inductive bias is mismatched** with task requirements.

---

## What We Built

### Models
| File | Description |
|------|-------------|
| `proposed_model.py` | Full Geodesic-Delta + mHC |
| `proposed_model_v2.py` | V2 with normalized generators (prevents collapse) |
| `proposed_model_mhc.py` | mHC mixing only (no rotation) |
| `model.py` | Baseline Transformer |

### Tasks
| Task | Dataset | Purpose |
|------|---------|---------|
| Grokking | `data/grokking/` | Modular arithmetic |
| Erasure | `data/erasure/` | Negation/correction |
| Isometry | `data/isometry/` | Pass-key retrieval |
| Reversibility | `data/reversibility/` | Path/rotation cancellation |
| **Rotation2D** | `data/rotation2d/` | TRUE geometric task |

### Diagnostic Tools
- `analyze_geodesic.py` - Checkpoint analysis for rotation generators
- Beta logging in `proposed_model.py` (disabled by default)

---

## Experimental Results Summary

### Grokking Task (Modular Arithmetic)
| Model | Val Loss | Notes |
|-------|----------|-------|
| Baseline | **1.58** | Winner |
| Geodesic | 1.60 | Gate closed (β≈0.0025) |
| mHC-only | 1.60 | No benefit |
| V2 (forced) | 2.32 | **+47% WORSE** |

### Key Finding: Rotation Generator Collapse
```
Layer 0: ||u|| = 0.000214, ||v|| = 0.000199
         ||A||_F = 0.000000  (rotation strength = ZERO)
         ||Qx - x|| = 0.000000  (no actual rotation)
```

The model doesn't just close the gate — it **zeros the rotation generators entirely**.

---

## Recommended Next Steps

### Option A: Test on TRUE Geometric Tasks (Highest Priority)
**Hypothesis**: Rotation helps when the task has actual geometric structure.

**Task: 2D Rotation Prediction** (already implemented in `data/rotation2d/`)
```
Input:  "1.0,0.0 0.0,1.0 -> 0.0,1.0 -1.0,0.0 = R90"
        (Original points → Rotated points → What angle?)
```

**Run Commands** (without torch.compile for speed):
```bash
# Baseline
python train.py config/train_rot2d_baseline_nocompile.py

# Geodesic  
python train_geodesic.py config/train_rot2d_geodesic_nocompile.py

# mHC-only
python train_mhc.py config/train_rot2d_mhc.py
```

**Expected Results**:
- If Geodesic wins → Theory validated for geometric tasks
- If Geodesic loses → Theory fundamentally flawed

### Option B: Architectural Pivot - Value-Path Rotation
**Rationale**: Rotation in residual stream disrupts gradient flow.

```python
# Current (problematic):
x_rotated = rotate(x)
x = x_rotated + attention(x_rotated)

# Proposed:
v_rotated = rotate(V)  # Rotate attention values only
attn_out = softmax(QK^T) @ v_rotated
x = x + attn_out  # Clean residual
```

**Implementation**: Modify `CausalSelfAttention` to apply rotation to V only.

### Option C: mHC-Only Investigation
**Question**: Do the mixing matrices alone provide benefit?

Already implemented in `proposed_model_mhc.py`. Preliminary results:
- Grokking: Val loss 1.60 (same as baseline)
- Rotation2D: Val loss 0.57 (need baseline comparison)

### Option D: Theoretical Analysis
**Questions to answer**:
1. What loss landscape properties make rotation useless?
2. Under what data distributions does orthogonal transform help?
3. Can we derive necessary conditions for geometric bias utility?

### Option E: Per-Head or Token-Specific Rotation
**Idea**: Instead of one global rotation, apply different rotations per attention head or per token.

```python
# Token-specific rotations
Q_per_token = self.rotation_table[token_ids]  # (B, S, n, n)
x_rotated = einsum(Q_per_token, x_streams)
```

---

## Technical Notes

### torch.compile Issues
When running multiple experiments in parallel with `compile=True`:
- Compilation spawns 32+ worker processes per script
- CPU becomes bottleneck (100% usage)
- GPU sits idle during compilation

**Solution**: Use `compile=False` for faster iteration, or run sequentially.

### GPU Memory
A100 80GB can easily handle these small models (1-10M params).
Memory is not the bottleneck — CPU compilation is.

### Logging Performance Impact
Beta logging with `random.random()` causes torch.compile graph breaks.
Disable logging for production runs: `ENABLE_BETA_LOGGING = False`

---

## File Structure

```
edelta/
├── model.py                    # Baseline Transformer
├── proposed_model.py           # Geodesic-Delta + mHC
├── proposed_model_v2.py        # V2 with normalized generators
├── proposed_model_mhc.py       # mHC-only (no rotation)
├── train.py                    # Baseline training script
├── train_geodesic.py           # Geodesic training script
├── train_geodesic_v2.py        # V2 training script
├── train_mhc.py                # mHC training script
├── analyze_geodesic.py         # Checkpoint analysis tool
├── configurator.py             # Config loading
├── data/
│   ├── grokking/prepare.py     # Modular arithmetic
│   ├── erasure/prepare.py      # Negation task
│   ├── isometry/prepare.py     # Pass-key task
│   ├── reversibility/prepare.py # Cancellation task
│   └── rotation2d/prepare.py   # TRUE geometric task
├── config/
│   ├── train_grok_*.py         # Grokking configs
│   ├── train_rot2d_*.py        # Rotation2D configs
│   └── ...
├── RESULTS.md                  # Original experiment results
├── DIAGNOSTIC_FINDINGS.md      # Deep diagnostic analysis
└── NEXT_STEPS.md              # This file
```

---

## Conclusions

### What We Learned
1. **Gradient descent is "lazy"**: If a simpler path exists, the optimizer takes it
2. **Inductive bias must match task**: Rotation doesn't help modular arithmetic
3. **Models can disable mechanisms**: u,v→0 is a clever way to become baseline
4. **Negative results are valuable**: Now we know when NOT to use geometric bias

### Open Questions
1. Does rotation help on TRUE geometric tasks (2D/3D)?
2. Can value-path rotation preserve benefits while keeping gradients clean?
3. What's the minimal task complexity where rotation becomes useful?

### Recommended Immediate Action
Run the Rotation2D experiment with `compile=False`:
```bash
cd /root/edelta
python train.py config/train_rot2d_baseline_nocompile.py 2>&1 | tee rot2d_base.log &
python train_geodesic.py config/train_rot2d_geodesic_nocompile.py 2>&1 | tee rot2d_geo.log &
wait
```

This will definitively answer: **Does rotation help when the task IS geometric?**

---

## References

- Original paper: "The Geodesic Manifold-Delta Transformer" (Shahmansoori, 2026)
- Deep Delta Learning: arXiv:2601.00417v1
- DeepSeek mHC: arXiv:2512.24880v1
- Illusion of Insight: arXiv:2601.00514v1
