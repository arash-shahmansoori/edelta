"""
Comparative Study Configuration

This configuration defines the "Fair Fight" parameters for comparing:
- Standard GPT (Baseline)
- Deep Delta Learning (DDL) - arXiv:2601.00417
- DeepSeek mHC (Baseline) - arXiv:2512.24880
- E∆-MHC-Geo (Proposed)

These parameters ensure scientific rigor by:
1. Matching parameter counts approximately across models
2. Using aggressive LR to expose instabilities
3. Disabling dropout for pure geometric precision testing
4. Using small-enough data to prevent brute-force memorization

Usage:
    python train_continuous.py --model_type gpt2 --dataset gyroscope
"""

# =============================================================================
# 1. Experiment Output
# =============================================================================
out_dir = 'out-comparative'
eval_interval = 100          # Evaluate every N steps
eval_iters = 40              # Number of batches for evaluation
log_interval = 50            # Log training loss every N steps
always_save_checkpoint = True

# =============================================================================
# 2. Data Settings
# =============================================================================
dataset = 'gyroscope'
data_dir = 'data'
batch_size = 64

# =============================================================================
# 3. Model Architecture ("Fair Fight" Configuration)
# =============================================================================
# All models use IDENTICAL structural parameters for fair comparison
# Only the INTERNAL MECHANISM differs (residual vs DDL vs mHC vs Cayley)

n_layer = 6       # Deep enough to show drift/collapse (6 layers)
n_head = 4        # Standard split (32 dim per head at n_embd=128)
n_embd = 128      # Width: Large enough for vectors, small to prevent memorization
n_streams = 4     # For mHC and E∆-MHC-Geo (matches n_head for symmetry)
dropout = 0.0     # CRUCIAL: No noise for regression. We need geometric precision.
bias = False      # Clean signal path (modern practice: LLaMA, PaLM)

# =============================================================================
# 4. Optimization
# =============================================================================
learning_rate = 1e-3    # Aggressive LR: Tests stability. E∆-MHC-Geo handles; DDL may crash
max_iters = 2000        # If method works, it converges by step 2000
min_lr = 1e-4           # Final LR after cosine decay

# Stabilizers
weight_decay = 0.1      # Standard L2 regularization
grad_clip = 1.0         # DDL needs this; E∆-MHC-Geo doesn't but keep for fairness

# =============================================================================
# 5. E∆-MHC-Geo Specific
# =============================================================================
# Gate regularization: L_gate = 4γ(1-γ) forces γ → 0 or 1 (avoid midpoint collapse)
gate_reg_weight = 0.01

# =============================================================================
# 6. Hardware
# =============================================================================
device = 'cuda'
compile = False  # Set True if PyTorch 2.0+ available
seed = 42

# =============================================================================
# 7. Dataset Specifications (EXACT from comparative study table)
# =============================================================================
DATASET_CONFIGS = {
    'gyroscope': {
        'input_dim': 16,
        'seq_len': 256,
        'n_train': 9000,
        'n_val': 1000,
        'metric': 'mse',
        'description': 'Continuous rotation prediction (tests manifold precision)',
        'target_weakness': 'DDL breaks at θ > 0.5 radians',
    },
    'correction': {
        'input_dim': 32,
        'seq_len': 32,
        'n_train': 4500,
        'n_val': 500,
        'metric': 'cosine_similarity',
        'description': 'Belief flip / negation (tests topological completeness)',
        'target_weakness': 'Cayley cannot achieve eigenvalue -1',
    },
    'stability': {
        'input_dim': 64,
        'seq_len_train': 128,
        'seq_len_test': 10000,
        'n_train': 900,
        'n_val': 100,
        'metric': 'norm_drift',
        'description': 'Long-horizon identity (tests unconditional isometry)',
        'target_weakness': 'Standard residuals accumulate norm drift',
    },
}

# =============================================================================
# 8. Expected Results (Hypotheses for Each Dataset)
# =============================================================================
EXPECTED_RESULTS = """
================================================================================
GYROSCOPE DATASET (Rotation Prediction)
================================================================================
Task: Predict v_{t+1} = R @ v_t where R is a rotation matrix
Metric: MSE Loss

Expected Performance:
    Standard GPT  : Loss plateaus ~0.1 (linear x + δ cannot represent rotation)
    DDL           : Unstable for large angles θ > 0.5, may diverge
    mHC           : Higher loss due to spectral dampening (doubly stochastic)
    E∆-MHC-Geo    : Near-zero loss across ALL angles (Cayley is exact rotation)

Why E∆-MHC-Geo Wins:
    - Cayley transform: Q = (I - A)(I + A)^{-1} is EXACTLY in SO(n)
    - Can rotate 179° in a single step with zero approximation error
    - DDL's linear update: x' = x - β(k·x)k cannot stay on the manifold

================================================================================
CORRECTION DATASET (Negation)
================================================================================
Task: Learn signal → flip sign (x → -x)
Metric: Cosine Similarity at flip point

Expected Performance:
    Standard GPT  : Struggles to learn x → -x (requires large weights)
    DDL           : Can negate at β=2, but training path is unstable
    mHC           : Cannot flip instantly (no eigenvalue -1 in doubly stochastic)
    E∆-MHC-Geo    : INSTANT flip via Householder (γ→0, β=2)

Why E∆-MHC-Geo Wins:
    - Householder reflection H = I - 2kkᵀ achieves EXACT negation
    - Thermodynamic gate detects "confusion" (signal) and switches to reflection
    - mHC's stochastic matrices have |λ| ≤ 1, cannot achieve λ = -1

================================================================================
STABILITY DATASET (Norm Preservation)
================================================================================
Task: Maintain ||v|| = 1 over 10,000 autoregressive steps
Metric: Norm drift (should be 0 for isometric methods)

Expected Performance:
    Standard GPT  : Norm drifts (grows or shrinks) after ~500 steps
    DDL           : Norm drifts because β ≠ 2 during training
    mHC           : Spectral collapse (streams converge to mean)
    E∆-MHC-Geo    : ||v|| = 1.0 for 10,000+ steps (PERFECT)

Why E∆-MHC-Geo Wins:
    - Cayley: ||Q @ x|| = ||x|| (orthogonal, eigenvalues on unit circle)
    - Householder at β=2: ||H @ x|| = ||x|| (orthogonal reflection)
    - Hybrid gate forces binary decisions, avoiding midpoint non-orthogonality
    - DDL is orthogonal ONLY at β ∈ {0, 2}, drifts otherwise
"""

# =============================================================================
# 9. Parameter Count Fairness Check
# =============================================================================
def estimate_params(n_layer, n_embd, n_head, n_streams=4):
    """Estimate parameter count for each model type."""
    # Attention: QKV projection + output projection
    attn = 4 * n_embd * n_embd
    # MLP: expand 4x then contract
    mlp = 8 * n_embd * n_embd
    # LayerNorms
    ln = 4 * n_embd
    
    # Per-block parameters
    base_block = attn + mlp + ln
    
    # Model-specific additions (approximate)
    gpt2_per_block = base_block
    ddl_per_block = base_block + 2 * (n_embd * n_embd // 4 + n_embd)  # k_net, beta_net
    mhc_per_block = base_block + 2 * n_streams * n_streams + 4 * n_streams  # H_res, H_pre, H_post
    edelta_per_block = base_block + 4 * n_embd + 2 * n_streams * n_streams  # Similar to mHC + gate
    
    return {
        'gpt2': n_layer * gpt2_per_block,
        'ddl': n_layer * ddl_per_block,
        'mhc': n_layer * mhc_per_block,
        'edelta': n_layer * edelta_per_block,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("COMPARATIVE STUDY CONFIGURATION")
    print("=" * 70)
    print()
    print("Fair Fight Model Parameters:")
    print(f"  n_layer     = {n_layer}")
    print(f"  n_embd      = {n_embd}")
    print(f"  n_head      = {n_head}")
    print(f"  n_streams   = {n_streams}")
    print(f"  dropout     = {dropout}")
    print(f"  bias        = {bias}")
    print()
    print("Training Parameters:")
    print(f"  batch_size  = {batch_size}")
    print(f"  max_iters   = {max_iters}")
    print(f"  lr          = {learning_rate}")
    print(f"  weight_decay= {weight_decay}")
    print(f"  grad_clip   = {grad_clip}")
    print()
    print("Dataset Specifications:")
    print("-" * 70)
    print(f"{'Dataset':<12} {'Train':<8} {'Val':<6} {'SeqLen':<8} {'Dim':<5} {'Metric'}")
    print("-" * 70)
    for name, cfg in DATASET_CONFIGS.items():
        seq_len = cfg.get('seq_len', f"{cfg.get('seq_len_train', '?')}/{cfg.get('seq_len_test', '?')}")
        print(f"{name:<12} {cfg['n_train']:<8} {cfg['n_val']:<6} {seq_len:<8} {cfg['input_dim']:<5} {cfg['metric']}")
    print("-" * 70)
    print()
    print("Estimated Parameter Counts (core transformer only):")
    params = estimate_params(n_layer, n_embd, n_head, n_streams)
    for model, count in params.items():
        print(f"  {model:<8}: ~{count/1e6:.2f}M parameters")
    print()
    print(EXPECTED_RESULTS)
