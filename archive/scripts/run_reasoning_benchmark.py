#!/usr/bin/env python3
"""
Text-Based Reasoning Shift Benchmark

Follows "The Illusion of Insight in Reasoning Models" (arXiv:2601.00514v1):
- Section 3: Shift detection methodology
- Section 4: Three complementary reasoning domains
- Section 5: Metrics (P(S), P(✓|S=1), P(✓|S=0))

This benchmark evaluates whether different transformer architectures can:
1. Learn to recognize correction signals in text
2. Successfully revise predictions after seeing correction cues
3. Generalize shift handling to new problems

Models compared:
- GPT2 (baseline): Standard residual connections
- DDL: Deep Delta Learning with rank-1 perturbation
- mHC: DeepSeek manifold Hyper-Connections
- Proposed (E∆-MHC-Geo): Hybrid Cayley-Householder with adaptive gating

Reference: https://arxiv.org/html/2601.00514v1
"""

import os
import sys
import argparse
import time
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


# =============================================================================
# SHIFT DETECTION (Following Paper Table 10)
# =============================================================================

# Lexical cues from paper's Table 10 (Appendix)
SHIFT_LEXICAL_CUES = {
    'reconsideration': [
        'wait', 'actually', 'hold on', 'let me reconsider', 
        'let me think again', 'on second thought', 'hmm'
    ],
    'negation': [
        'no,', 'no ', 'that\'s wrong', 'incorrect', 'mistake',
        'i was wrong', 'that\'s not right'
    ],
    'revision': [
        'the correct answer is', 'i should have', 'let me redo',
        'the real answer', 'correction:', 'revised:'
    ],
    'backtracking': [
        'going back', 'returning to', 'starting over',
        'let me start fresh', 'from the beginning'
    ],
}


def detect_lexical_shifts(text: str) -> Dict:
    """
    Detect reasoning shifts using lexical cues (paper Table 10).
    
    Returns dict with:
    - has_shift: bool
    - shift_positions: list of (position, cue, category)
    - shift_count: int
    """
    text_lower = text.lower()
    shifts = []
    
    for category, cues in SHIFT_LEXICAL_CUES.items():
        for cue in cues:
            pos = 0
            while True:
                idx = text_lower.find(cue, pos)
                if idx == -1:
                    break
                shifts.append({
                    'position': idx,
                    'cue': cue,
                    'category': category,
                    'context': text[max(0, idx-15):idx+len(cue)+15]
                })
                pos = idx + 1
    
    # Sort by position
    shifts.sort(key=lambda x: x['position'])
    
    return {
        'has_shift': len(shifts) > 0,
        'shift_count': len(shifts),
        'shifts': shifts,
        'categories': list(set(s['category'] for s in shifts))
    }


def extract_final_answer(text: str) -> Optional[int]:
    """Extract numerical answer from text."""
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*(-?\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Look for final number after = sign
    matches = re.findall(r'=\s*(-?\d+)', text)
    if matches:
        return int(matches[-1])
    
    # Look for any final number
    matches = re.findall(r'(-?\d+)', text)
    if matches:
        return int(matches[-1])
    
    return None


# =============================================================================
# MODEL LOADING
# =============================================================================

def create_model(model_name: str, config_override: Dict = None):
    """Create model by name with consistent config."""
    
    # Base configuration
    base_config = {
        'vocab_size': 256,  # Character-level for simplicity
        'block_size': 256,
        'n_layer': 6,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.1,
        'bias': True,
    }
    
    if config_override:
        base_config.update(config_override)
    
    if model_name == 'gpt2':
        from model import GPT, GPTConfig
        config = GPTConfig(**base_config)
        return GPT(config), 'GPT2 (baseline)'
        
    elif model_name == 'ddl':
        from proposed_model_ddl import GPT as DDLGPT, GPTConfig as DDLConfig
        config = DDLConfig(**base_config)
        return DDLGPT(config), 'DDL (Deep Delta Learning)'
        
    elif model_name == 'mhc':
        from proposed_model_mhc_real import GPT as mHCGPT, GPTConfig as mHCConfig
        config = mHCConfig(**base_config)
        return mHCGPT(config), 'mHC (DeepSeek)'
        
    elif model_name == 'proposed':
        from proposed_model_hybrid import GPT as HybridGPT, GPTConfig as HybridConfig
        config = HybridConfig(**base_config)
        return HybridGPT(config), 'E∆-MHC-Geo (Proposed)'
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_reasoning_data(
    data_dir: str = 'data/reasoning_shift',
    max_length: int = 256,
    device: str = 'cuda'
) -> Dict:
    """
    Prepare tokenized reasoning data for training.
    
    Uses character-level tokenization for simplicity.
    """
    
    # Generate dataset if not exists
    if not os.path.exists(data_dir):
        flush_print(f"Generating reasoning dataset in {data_dir}...")
        from data.reasoning_shift import generate_reasoning_dataset
        generate_reasoning_dataset(
            n_samples=2000,
            shift_ratio=0.5,
            output_dir=data_dir
        )
    
    # Load text data
    with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
        train_text = f.read()
    
    with open(os.path.join(data_dir, 'val.txt'), 'r') as f:
        val_text = f.read()
    
    # Load problem metadata
    train_problems = np.load(os.path.join(data_dir, 'train_problems.npy'), allow_pickle=True)
    val_problems = np.load(os.path.join(data_dir, 'val_problems.npy'), allow_pickle=True)
    
    # Character-level encoding
    chars = sorted(list(set(train_text + val_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)
    
    flush_print(f"Vocabulary size: {vocab_size}")
    
    def encode(text):
        return [stoi.get(c, 0) for c in text]
    
    def decode(tokens):
        return ''.join([itos.get(t, '?') for t in tokens])
    
    # Prepare sequences from problems
    def prepare_sequences(problems, max_len):
        sequences = []
        labels = []
        metadata = []
        
        for prob in problems:
            text = prob['sequence']
            tokens = encode(text)
            
            if len(tokens) < 10:  # Skip too short
                continue
            
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [0] * (max_len - len(tokens))
            
            # Language modeling: predict next token
            x = tokens[:-1]
            y = tokens[1:]
            
            sequences.append(x)
            labels.append(y)
            metadata.append({
                'has_shift': prob['has_shift'],
                'type': prob['type'],
                'correct_answer': prob['segments']['correct_answer'],
            })
        
        return (
            torch.tensor(sequences, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device),
            metadata
        )
    
    train_x, train_y, train_meta = prepare_sequences(train_problems, max_length)
    val_x, val_y, val_meta = prepare_sequences(val_problems, max_length)
    
    flush_print(f"Train: {len(train_x)} sequences, Val: {len(val_x)} sequences")
    
    return {
        'train_x': train_x,
        'train_y': train_y,
        'train_meta': train_meta,
        'val_x': val_x,
        'val_y': val_y,
        'val_meta': val_meta,
        'vocab_size': vocab_size,
        'encode': encode,
        'decode': decode,
        'stoi': stoi,
        'itos': itos,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    data: Dict,
    max_iters: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    eval_interval: int = 100,
    device: str = 'cuda',
    model_name: str = 'model',
) -> Dict:
    """Train model on reasoning dataset."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    
    n_train = len(train_x)
    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
    
    flush_print(f"\nTraining {model_name}...")
    flush_print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    
    for iter_num in range(max_iters):
        model.train()
        
        # Sample batch
        idx = torch.randint(0, n_train, (batch_size,))
        xb, yb = train_x[idx], train_y[idx]
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['train_loss'].append(loss.item())
        
        # Evaluate periodically
        if (iter_num + 1) % eval_interval == 0 or iter_num == 0:
            model.eval()
            with torch.no_grad():
                # Compute validation loss
                val_logits, val_loss = model(val_x[:500], val_y[:500])
                perplexity = torch.exp(val_loss).item()
                
                history['val_loss'].append(val_loss.item())
                history['val_perplexity'].append(perplexity)
                
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                
                flush_print(f"  Step {iter_num+1:4d}: train_loss={loss.item():.4f}, "
                           f"val_loss={val_loss.item():.4f}, perplexity={perplexity:.2f}")
    
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'final_perplexity': history['val_perplexity'][-1] if history['val_perplexity'] else float('inf'),
    }


# =============================================================================
# EVALUATION WITH SHIFT ANALYSIS
# =============================================================================

@torch.no_grad()
def generate_completion(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
) -> torch.Tensor:
    """Generate completion from model."""
    
    model.eval()
    idx = prompt_tokens.clone()
    
    for _ in range(max_new_tokens):
        # Crop to block size if needed
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        
        # Get logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
        
        # Stop at newline or answer
        # (simplified stopping condition)
    
    return idx


def evaluate_shift_handling(
    model: nn.Module,
    data: Dict,
    n_samples: int = 100,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate model's shift handling following paper metrics.
    
    Returns:
        Dict with paper-style metrics:
        - P(S): Shift prevalence in model outputs
        - P(✓|S=1): Accuracy when shift detected
        - P(✓|S=0): Accuracy when no shift detected
        - shift_benefit: P(✓|S=1) - P(✓|S=0)
    """
    
    model.eval()
    
    val_x = data['val_x'][:n_samples]
    val_meta = data['val_meta'][:n_samples]
    decode = data['decode']
    
    results = {
        'shift_samples': [],
        'no_shift_samples': [],
        'total_correct': 0,
        'total': 0,
    }
    
    # For each validation sample, check if model output contains shift cues
    # and whether it gets the correct answer
    for i in range(len(val_x)):
        # Get model's prediction for the sequence
        with torch.no_grad():
            logits, loss = model(val_x[i:i+1], data['val_y'][i:i+1])
        
        # Get predicted tokens
        pred_tokens = logits.argmax(dim=-1)[0].cpu().numpy()
        pred_text = decode(pred_tokens.tolist())
        
        # Detect shifts in prediction
        shift_result = detect_lexical_shifts(pred_text)
        
        # Extract predicted answer
        pred_answer = extract_final_answer(pred_text)
        correct_answer = val_meta[i]['correct_answer']
        is_correct = (pred_answer == correct_answer) if pred_answer is not None else False
        
        # Ground truth: does this sample have a shift in training data?
        gt_has_shift = val_meta[i]['has_shift']
        
        sample_result = {
            'idx': i,
            'predicted_shift': shift_result['has_shift'],
            'gt_has_shift': gt_has_shift,
            'pred_answer': pred_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'loss': loss.item(),
        }
        
        if shift_result['has_shift']:
            results['shift_samples'].append(sample_result)
        else:
            results['no_shift_samples'].append(sample_result)
        
        results['total'] += 1
        if is_correct:
            results['total_correct'] += 1
    
    # Compute paper metrics
    n_shift = len(results['shift_samples'])
    n_no_shift = len(results['no_shift_samples'])
    
    shift_correct = sum(1 for s in results['shift_samples'] if s['is_correct'])
    no_shift_correct = sum(1 for s in results['no_shift_samples'] if s['is_correct'])
    
    metrics = {
        'P(S)': n_shift / results['total'] if results['total'] > 0 else 0,
        'P(correct|S=1)': shift_correct / n_shift if n_shift > 0 else 0,
        'P(correct|S=0)': no_shift_correct / n_no_shift if n_no_shift > 0 else 0,
        'overall_accuracy': results['total_correct'] / results['total'] if results['total'] > 0 else 0,
        'n_shift': n_shift,
        'n_no_shift': n_no_shift,
        'n_total': results['total'],
    }
    
    # Shift benefit (paper's key metric)
    # Negative means shifts are harmful (paper's finding)
    metrics['shift_benefit'] = metrics['P(correct|S=1)'] - metrics['P(correct|S=0)']
    
    return metrics


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(args):
    """Run complete reasoning shift benchmark."""
    
    flush_print("="*80)
    flush_print("REASONING SHIFT BENCHMARK")
    flush_print("Following 'The Illusion of Insight in Reasoning Models' (arXiv:2601.00514v1)")
    flush_print("="*80)
    
    # Prepare data
    flush_print("\n[1] Preparing reasoning dataset...")
    data = prepare_reasoning_data(
        data_dir=args.data_dir,
        max_length=args.max_length,
        device=args.device
    )
    
    # Update vocab size in config
    config_override = {'vocab_size': data['vocab_size']}
    
    # Models to compare
    models_to_test = ['gpt2', 'ddl', 'mhc', 'proposed']
    
    all_results = {}
    
    flush_print("\n[2] Training and evaluating models...")
    
    for model_name in models_to_test:
        flush_print(f"\n{'='*60}")
        
        try:
            model, description = create_model(model_name, config_override)
            flush_print(f"Model: {description}")
            
            # Train
            train_result = train_model(
                model=model,
                data=data,
                max_iters=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_interval=args.eval_interval,
                device=args.device,
                model_name=description,
            )
            
            # Evaluate shift handling
            flush_print(f"\n  Evaluating shift handling...")
            shift_metrics = evaluate_shift_handling(
                model=train_result['model'],
                data=data,
                n_samples=min(args.n_eval, len(data['val_x'])),
                device=args.device,
            )
            
            all_results[model_name] = {
                'description': description,
                'train_result': train_result,
                'shift_metrics': shift_metrics,
            }
            
            flush_print(f"\n  Results for {description}:")
            flush_print(f"    Best Val Loss:      {train_result['best_val_loss']:.4f}")
            flush_print(f"    Final Perplexity:   {train_result['final_perplexity']:.2f}")
            flush_print(f"    P(S) Shift Prevalence: {shift_metrics['P(S)']:.4f}")
            flush_print(f"    P(✓|S=1):           {shift_metrics['P(correct|S=1)']:.4f}")
            flush_print(f"    P(✓|S=0):           {shift_metrics['P(correct|S=0)']:.4f}")
            flush_print(f"    Shift Benefit:      {shift_metrics['shift_benefit']:+.4f}")
            
        except Exception as e:
            flush_print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # Summary comparison
    flush_print("\n" + "="*80)
    flush_print("SUMMARY: REASONING SHIFT BENCHMARK RESULTS")
    flush_print("="*80)
    
    flush_print("\n" + "-"*80)
    flush_print(f"{'Model':<25} {'Val Loss':>10} {'Perplexity':>12} {'P(S)':>8} {'P(✓|S=1)':>10} {'P(✓|S=0)':>10} {'Benefit':>10}")
    flush_print("-"*80)
    
    for model_name, result in all_results.items():
        if 'error' in result:
            flush_print(f"{model_name:<25} ERROR: {result['error']}")
            continue
        
        desc = result['description'][:24]
        val_loss = result['train_result']['best_val_loss']
        perplexity = result['train_result']['final_perplexity']
        metrics = result['shift_metrics']
        
        flush_print(f"{desc:<25} {val_loss:>10.4f} {perplexity:>12.2f} "
                   f"{metrics['P(S)']:>8.4f} {metrics['P(correct|S=1)']:>10.4f} "
                   f"{metrics['P(correct|S=0)']:>10.4f} {metrics['shift_benefit']:>+10.4f}")
    
    flush_print("-"*80)
    
    # Analysis
    flush_print("\n" + "="*80)
    flush_print("ANALYSIS (Following Paper Methodology)")
    flush_print("="*80)
    
    flush_print("""
Key Metrics from "The Illusion of Insight" (arXiv:2601.00514v1):

1. P(S) - Shift Prevalence: 
   How often the model produces reasoning shifts.
   Paper finding: ~6.31% in DeepSeek-R1, much lower than perceived.

2. P(✓|S=1) - Accuracy Given Shift:
   When a shift occurs, how often is the final answer correct?
   Paper finding: 2.57% - shifts usually indicate confusion, not insight.

3. P(✓|S=0) - Accuracy Given No Shift:
   When no shift occurs, how often is the answer correct?
   Paper finding: 16.44% - stable reasoning is generally better.

4. Shift Benefit = P(✓|S=1) - P(✓|S=0):
   NEGATIVE means shifts are harmful (paper's main finding).
   Paper finding: -13.87pp, showing "Aha!" moments are often illusory.

For geometric models (DDL, Proposed):
- If they show HIGHER P(✓|S=1), they may better handle belief revision.
- If their shift_benefit is LESS NEGATIVE, they handle shifts more gracefully.
""")
    
    # Determine best models
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if valid_results:
        # Best by perplexity
        best_perplexity = min(valid_results.items(), 
                             key=lambda x: x[1]['train_result']['final_perplexity'])
        flush_print(f"\nBest Perplexity: {best_perplexity[1]['description']} "
                   f"({best_perplexity[1]['train_result']['final_perplexity']:.2f})")
        
        # Best by shift benefit
        best_shift = max(valid_results.items(),
                        key=lambda x: x[1]['shift_metrics']['shift_benefit'])
        flush_print(f"Best Shift Handling: {best_shift[1]['description']} "
                   f"(benefit: {best_shift[1]['shift_metrics']['shift_benefit']:+.4f})")
        
        # Geometric vs baseline comparison
        geo_models = ['ddl', 'proposed']
        baseline_models = ['gpt2']
        
        geo_benefit = np.mean([valid_results[m]['shift_metrics']['shift_benefit'] 
                              for m in geo_models if m in valid_results])
        baseline_benefit = np.mean([valid_results[m]['shift_metrics']['shift_benefit'] 
                                   for m in baseline_models if m in valid_results])
        
        flush_print(f"\nGeometric models avg shift_benefit: {geo_benefit:+.4f}")
        flush_print(f"Baseline (GPT2) shift_benefit:      {baseline_benefit:+.4f}")
        
        if geo_benefit > baseline_benefit:
            flush_print("\n✓ Geometric models handle reasoning shifts better than baseline!")
        else:
            flush_print("\n⚠ No clear advantage for geometric models on shift handling.")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Reasoning Shift Benchmark')
    
    parser.add_argument('--data_dir', type=str, default='data/reasoning_shift',
                       help='Directory for reasoning dataset')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--max_iters', type=int, default=1000,
                       help='Training iterations per model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=100,
                       help='Evaluation interval')
    parser.add_argument('--n_eval', type=int, default=200,
                       help='Number of samples for shift evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    results = run_benchmark(args)
    
    flush_print("\n" + "="*80)
    flush_print("Benchmark complete!")
    flush_print("="*80)


if __name__ == '__main__':
    main()
