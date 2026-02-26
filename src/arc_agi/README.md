# ARC-AGI Experiments: E∆ vs JPmHC on TRM Backbone

## Overview

This directory contains mixer modules for integrating E∆ and JPmHC
into the Tiny Recursive Model (TRM) backbone for ARC-AGI evaluation.

## Architecture

Both mixers wrap the attention and FFN sub-blocks in each TRM layer,
following the parallel routing topology from JPmHC (Eq. 14):

```
x_out = H_res · x_streams + H_post · (F(avg(H_pre · x_streams)) ⊗ 1_n)
```

Three variants:
- **Baseline TRM**: Standard residual connections (no mixer)
- **TRM + JPmHC**: Iterative Cayley retraction for H_res (SO(n) only)
- **TRM + E∆**: Exact Cayley + Householder + gate for H_res (full O(n))

## Files

- `mixers.py` — JPmHCMixer and EdeltaMixer implementations
- `trm_block.py` — Modified TRM block with pluggable mixer support

## Integration with TRM

To use with the TRM codebase (https://github.com/SamsungSAILMontreal/TinyRecursiveModels):

1. Clone TRM repo alongside this project
2. Replace `TinyRecursiveReasoningModel_ACTV1Block` with `MixedTRMBlock`
3. Add `mixer_type` config option ('none', 'jpmhc', 'edelta')

The mixer modules handle:
- Multi-stream reshaping (n_embd → n_streams × d_stream)
- Dynamic routing matrix generation (H_pre, H_post)
- Orthogonal residual mixing (H_res via Cayley/Householder)
- Per-stream attention and FFN evaluation

## Config (matching JPmHC paper Table 10)

```yaml
hidden_size: 512      # = n_streams × d_stream = 4 × 128
n_streams: 4
num_heads: 8          # 8 heads at d_stream=128 → 16 per head
H_cycles: 3
L_cycles: 6
L_layers: 2
halt_max_steps: 16
```

## Expected Metrics

- Exact-match accuracy (greedy)
- Pass@k (k = 1, 10, 100)
- Per-task gate analysis (γ values for E∆)
- Learning curves at regular checkpoints
- Wall-clock training time
```
