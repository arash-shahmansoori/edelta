"""
FULL GPT LANGUAGE TASK COMPARISON: DDC vs DDL vs Baseline

Realistic tasks designed to showcase DDC's theoretical advantages:

1. CORRECTION TASK - "Wait, I meant..." - tests negation capability
   - DDC-Hybrid should excel (rotation + reflection)
   - DDL should also work (Householder reflection)
   - Pure DDC may struggle (cannot negate)

2. MULTI-HOP REASONING - "A→B, B→C, Actually B→D, What is A?"
   - Tests information preservation + selective negation
   - DDC-Hybrid: preserve A→B (rotate), negate B→C (reflect)

3. DEEP LANGUAGE MODELING - Shakespeare at extreme depth
   - Tests gradient stability in very deep networks
   - DDC's orthogonality should help

Uses FULL GPT models with:
- Causal Self-Attention
- Token/Position Embeddings
- MLP layers
- DDC/DDL operators
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Ensure we can import the models
sys.path.insert(0, '/root/edelta')


# =============================================================================
# Task 1: Correction Task Dataset
# =============================================================================

def create_correction_dataset(n_samples=10000, vocab_size=100):
    """
    Task: Process instruction, then correction, output corrected result.
    
    Format: "Add 5 to X. X=10. Wait, subtract 5. Answer: 5"
    
    Why DDC-Hybrid should win:
    - Must PRESERVE initial value X (rotation preserves)
    - Must NEGATE initial operation (reflection negates)
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
    }
    corrections = [
        "Wait, I meant",
        "Actually, no",
        "Scratch that",
        "On second thought",
        "Never mind,",
    ]
    
    data = []
    for _ in range(n_samples):
        x = np.random.randint(1, 20)
        y = np.random.randint(1, 10)
        
        ops = list(operations.keys())
        op1 = ops[np.random.randint(len(ops))]
        op2 = ops[np.random.randint(len(ops))]
        
        correction = corrections[np.random.randint(len(corrections))]
        
        # The CORRECTED answer uses op2, not op1
        answer = operations[op2](x, y)
        
        text = f"Task: {op1} {y} to X. X={x}. {correction} {op2} {y}. Answer: {answer}"
        data.append(text)
    
    return data


# =============================================================================
# Task 2: Multi-Hop Reasoning with Corrections
# =============================================================================

def create_multihop_dataset(n_samples=10000):
    """
    Task: Multi-hop reasoning with mid-chain corrections.
    
    Format: "A is 5. B is A+3. Wait, B is A+7. C is B*2. What is C? 24"
    
    Why DDC-Hybrid should win:
    - Must PRESERVE A value through chain (rotation)
    - Must NEGATE B=A+3 when correction appears (reflection)
    - Must PROPAGATE correction to final answer
    """
    data = []
    for _ in range(n_samples):
        a = np.random.randint(1, 10)
        
        # Initial chain
        b_add_wrong = np.random.randint(1, 5)
        b_add_correct = np.random.randint(5, 10)  # Different!
        c_mult = np.random.randint(2, 4)
        
        # Corrected computation
        b_correct = a + b_add_correct
        c_correct = b_correct * c_mult
        
        text = f"A is {a}. B is A+{b_add_wrong}. Wait, B is A+{b_add_correct}. C is B*{c_mult}. What is C? {c_correct}"
        data.append(text)
    
    return data


# =============================================================================
# Task 3: Copy with Noise (Information Preservation)
# =============================================================================

def create_copy_with_noise_dataset(n_samples=10000, seq_len=20):
    """
    Task: Copy a sequence while ignoring noise markers.
    
    Format: "Copy: a b [NOISE] c d [NOISE] e. Output: a b c d e"
    
    Why DDC should help:
    - Must PRESERVE signal exactly (orthogonality maintains norms)
    - Must FILTER noise (selective processing)
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    data = []
    for _ in range(n_samples):
        # Generate clean sequence
        length = np.random.randint(3, seq_len // 2)
        clean = [alphabet[np.random.randint(len(alphabet))] for _ in range(length)]
        
        # Add noise markers at random positions
        noisy = []
        for char in clean:
            if np.random.random() < 0.3:
                noisy.append('[NOISE]')
            noisy.append(char)
        
        text = f"Copy: {' '.join(noisy)}. Output: {' '.join(clean)}"
        data.append(text)
    
    return data


# =============================================================================
# Tokenizer
# =============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(list(chars))
        
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 is padding
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.idx_to_char[0] = '<PAD>'
        self.vocab_size = len(chars) + 1
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join([self.idx_to_char.get(i, '?') for i in ids])


def prepare_data(texts, tokenizer, block_size=128):
    """Prepare data for training."""
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < block_size:
            ids = ids + [0] * (block_size - len(ids))
        else:
            ids = ids[:block_size]
        all_ids.append(ids)
    return np.array(all_ids, dtype=np.int64)


# =============================================================================
# Training Function for Full GPT Models
# =============================================================================

def train_full_gpt(model_class, config_class, train_data, val_data, 
                   config_kwargs, max_iters=3000, batch_size=32, lr=3e-4, 
                   device='cuda', model_name='model'):
    """Train a full GPT model and return metrics."""
    
    config = config_class(**config_kwargs)
    model = model_class(config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    n_params = sum(p.numel() for p in model.parameters())
    block_size = config_kwargs['block_size']
    
    train_data = torch.from_numpy(train_data).long().to(device)
    val_data = torch.from_numpy(val_data).long().to(device)
    
    def get_batch(data, batch_size):
        idx = torch.randint(len(data), (batch_size,))
        batch = data[idx]
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        return x, y
    
    # Metrics
    train_losses = []
    val_losses = []
    grad_norms = []
    
    best_val_loss = float('inf')
    
    t0 = time.time()
    
    for it in range(max_iters):
        model.train()
        x, y = get_batch(train_data, batch_size)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        if torch.isnan(loss) or total_norm > 1e6:
            print(f"    {model_name}: EXPLODED at iter {it}")
            return {
                'converged': False,
                'best_val_loss': float('inf'),
                'n_params': n_params,
            }
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate
        if it % 300 == 0:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0
                for _ in range(20):
                    x, y = get_batch(val_data, batch_size)
                    _, vloss = model(x, y)
                    val_loss_sum += vloss.item()
                val_loss = val_loss_sum / 20
                
                val_losses.append((it, val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            print(f"    {model_name:15s} iter {it:4d}: train={loss.item():.4f}, "
                  f"val={val_loss:.4f}, grad={total_norm:.2f}")
    
    return {
        'converged': True,
        'n_params': n_params,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'mean_grad_norm': np.mean(grad_norms[-500:]) if len(grad_norms) > 500 else np.mean(grad_norms),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'grad_norms': grad_norms,
        'train_time': time.time() - t0,
    }


# =============================================================================
# Main Comparison
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("FULL GPT LANGUAGE TASK COMPARISON")
    print("DDC vs DDL vs Baseline vs DDC-Hybrid")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Import models
    print("\nImporting models...")
    from model import GPTConfig as BaselineConfig, GPT as BaselineGPT
    from proposed_model_ddl import GPTConfig as DDLConfig, GPT as DDLGPT
    from proposed_model_ddc import GPTConfig as DDCConfig, GPT as DDCGPT
    from proposed_model_ddc_hybrid import GPTConfig as DDCHybridConfig, GPT as DDCHybridGPT
    
    # =========================================================================
    # TASK 1: CORRECTION TASK
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: CORRECTION TASK")
    print("Tests: Negation capability (DDC-Hybrid, DDL should excel)")
    print("=" * 80)
    
    print("\nCreating correction dataset...")
    correction_train = create_correction_dataset(8000)
    correction_val = create_correction_dataset(1000)
    
    tokenizer = CharTokenizer(correction_train + correction_val)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    train_data = prepare_data(correction_train, tokenizer, block_size=128)
    val_data = prepare_data(correction_val, tokenizer, block_size=128)
    print(f"  Train: {train_data.shape}, Val: {val_data.shape}")
    
    config_kwargs = {
        'block_size': 127,  # -1 for target shift
        'vocab_size': tokenizer.vocab_size,
        'n_layer': 6,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': False,
    }
    
    results_correction = {}
    
    # Baseline
    print("\n  Training Baseline...")
    results_correction['baseline'] = train_full_gpt(
        BaselineGPT, BaselineConfig, train_data, val_data, config_kwargs,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='Baseline'
    )
    torch.cuda.empty_cache()
    
    # DDL
    print("\n  Training DDL...")
    results_correction['ddl'] = train_full_gpt(
        DDLGPT, DDLConfig, train_data, val_data, config_kwargs,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDL'
    )
    torch.cuda.empty_cache()
    
    # DDC
    print("\n  Training DDC...")
    ddc_config = {**config_kwargs, 'n_streams': 4}
    results_correction['ddc'] = train_full_gpt(
        DDCGPT, DDCConfig, train_data, val_data, ddc_config,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC'
    )
    torch.cuda.empty_cache()
    
    # DDC-Hybrid
    print("\n  Training DDC-Hybrid...")
    results_correction['ddc_hybrid'] = train_full_gpt(
        DDCHybridGPT, DDCHybridConfig, train_data, val_data, ddc_config,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC-Hybrid'
    )
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TASK 2: MULTI-HOP REASONING
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 2: MULTI-HOP REASONING WITH CORRECTIONS")
    print("Tests: Information preservation + selective negation")
    print("=" * 80)
    
    print("\nCreating multi-hop dataset...")
    multihop_train = create_multihop_dataset(8000)
    multihop_val = create_multihop_dataset(1000)
    
    tokenizer2 = CharTokenizer(multihop_train + multihop_val)
    print(f"  Vocab size: {tokenizer2.vocab_size}")
    
    train_data2 = prepare_data(multihop_train, tokenizer2, block_size=128)
    val_data2 = prepare_data(multihop_val, tokenizer2, block_size=128)
    
    config_kwargs2 = {**config_kwargs, 'vocab_size': tokenizer2.vocab_size}
    ddc_config2 = {**config_kwargs2, 'n_streams': 4}
    
    results_multihop = {}
    
    # Baseline
    print("\n  Training Baseline...")
    results_multihop['baseline'] = train_full_gpt(
        BaselineGPT, BaselineConfig, train_data2, val_data2, config_kwargs2,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='Baseline'
    )
    torch.cuda.empty_cache()
    
    # DDL
    print("\n  Training DDL...")
    results_multihop['ddl'] = train_full_gpt(
        DDLGPT, DDLConfig, train_data2, val_data2, config_kwargs2,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDL'
    )
    torch.cuda.empty_cache()
    
    # DDC
    print("\n  Training DDC...")
    results_multihop['ddc'] = train_full_gpt(
        DDCGPT, DDCConfig, train_data2, val_data2, ddc_config2,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC'
    )
    torch.cuda.empty_cache()
    
    # DDC-Hybrid
    print("\n  Training DDC-Hybrid...")
    results_multihop['ddc_hybrid'] = train_full_gpt(
        DDCHybridGPT, DDCHybridConfig, train_data2, val_data2, ddc_config2,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC-Hybrid'
    )
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TASK 3: COPY WITH NOISE
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 3: COPY WITH NOISE (Information Preservation)")
    print("Tests: Signal preservation through noise")
    print("=" * 80)
    
    print("\nCreating copy-with-noise dataset...")
    copy_train = create_copy_with_noise_dataset(8000)
    copy_val = create_copy_with_noise_dataset(1000)
    
    tokenizer3 = CharTokenizer(copy_train + copy_val)
    print(f"  Vocab size: {tokenizer3.vocab_size}")
    
    train_data3 = prepare_data(copy_train, tokenizer3, block_size=128)
    val_data3 = prepare_data(copy_val, tokenizer3, block_size=128)
    
    config_kwargs3 = {**config_kwargs, 'vocab_size': tokenizer3.vocab_size}
    ddc_config3 = {**config_kwargs3, 'n_streams': 4}
    
    results_copy = {}
    
    # Baseline
    print("\n  Training Baseline...")
    results_copy['baseline'] = train_full_gpt(
        BaselineGPT, BaselineConfig, train_data3, val_data3, config_kwargs3,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='Baseline'
    )
    torch.cuda.empty_cache()
    
    # DDL
    print("\n  Training DDL...")
    results_copy['ddl'] = train_full_gpt(
        DDLGPT, DDLConfig, train_data3, val_data3, config_kwargs3,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDL'
    )
    torch.cuda.empty_cache()
    
    # DDC
    print("\n  Training DDC...")
    results_copy['ddc'] = train_full_gpt(
        DDCGPT, DDCConfig, train_data3, val_data3, ddc_config3,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC'
    )
    torch.cuda.empty_cache()
    
    # DDC-Hybrid
    print("\n  Training DDC-Hybrid...")
    results_copy['ddc_hybrid'] = train_full_gpt(
        DDCHybridGPT, DDCHybridConfig, train_data3, val_data3, ddc_config3,
        max_iters=2000, batch_size=32, lr=3e-4, device=device, model_name='DDC-Hybrid'
    )
    torch.cuda.empty_cache()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    all_results = {
        'Correction': results_correction,
        'Multi-Hop': results_multihop,
        'Copy-Noise': results_copy,
    }
    
    models = ['baseline', 'ddl', 'ddc', 'ddc_hybrid']
    
    for task_name, results in all_results.items():
        print(f"\n{task_name} Task:")
        print("-" * 60)
        print(f"{'Model':<15} {'Val Loss':<12} {'Grad Norm':<12} {'Time (s)':<10}")
        print("-" * 60)
        
        for m in models:
            r = results[m]
            if r['converged']:
                print(f"{m:<15} {r['best_val_loss']:<12.4f} {r['mean_grad_norm']:<12.2f} {r['train_time']:<10.1f}")
            else:
                print(f"{m:<15} {'FAILED':<12}")
        
        # Find winner
        converged = [(m, results[m]['best_val_loss']) for m in models if results[m]['converged']]
        if converged:
            winner = min(converged, key=lambda x: x[1])
            print(f"\n  WINNER: {winner[0]} (val_loss={winner[1]:.4f})")
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("""
EXPECTED PATTERNS:

1. CORRECTION TASK:
   - DDL should excel (Householder reflection can negate)
   - DDC-Hybrid should also excel (has reflection capability)
   - Pure DDC may struggle (cannot negate - rotation only)

2. MULTI-HOP REASONING:
   - DDC-Hybrid should win (needs BOTH preserve AND negate)
   - DDL might struggle (can negate but less structured)
   
3. COPY WITH NOISE:
   - DDC should help (orthogonality preserves signal)
   - Baseline might be fine (attention can learn to ignore noise)

KEY METRICS TO WATCH:
- Val loss shows task performance
- Gradient norm shows training stability
- DDC's advantage should appear in stability metrics
""")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'baseline': 'gray', 'ddl': 'red', 'ddc': 'blue', 'ddc_hybrid': 'green'}
    
    for idx, (task_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        losses = [results[m]['best_val_loss'] if results[m]['converged'] else 0 for m in models]
        bars = ax.bar(models, losses, color=[colors[m] for m in models])
        ax.set_title(f'{task_name} Task', fontsize=12)
        ax.set_ylabel('Validation Loss')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            if loss > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{loss:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('language_task_comparison.png', dpi=150)
    print(f"\nPlot saved to: language_task_comparison.png")


if __name__ == '__main__':
    run_comparison()
