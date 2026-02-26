"""
Parameter Counter Utility for E∆-MHC-Geo Models

Computes and verifies parameter counts across all model architectures
to ensure fair comparison in experiments.

Usage:
    uv run src/utils/param_counter.py
    uv run src/utils/param_counter.py --n_layer 6 --n_embd 128
    uv run src/utils/param_counter.py --find_match  # Find n_layer for baselines to match E∆
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from src.models.baseline_gpt import GPT as BaselineGPT, GPTConfig as BaselineConfig
from src.models.ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from src.models.mhc import GPT as mHCGPT, GPTConfig as mHCConfig
from src.models.jpmhc import GPT as JPmHCGPT, GPTConfig as JPmHCConfig
from src.models.edelta_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_params_by_component(model: nn.Module) -> dict:
    """Count parameters grouped by component name."""
    groups = {}
    for name, param in model.named_parameters():
        # Extract top-level component name
        parts = name.split('.')
        if len(parts) >= 2:
            component = parts[1] if parts[0] == 'transformer' else parts[0]
        else:
            component = parts[0]
        
        if component not in groups:
            groups[component] = 0
        groups[component] += param.numel()
    
    return dict(sorted(groups.items(), key=lambda x: -x[1]))


def create_model(model_type: str, n_layer: int, n_head: int, n_embd: int, 
                 n_streams: int = 4, block_size: int = 64, verbose: bool = False):
    """Create a model of the specified type."""
    
    if model_type == 'gpt':
        config = BaselineConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            dropout=0, bias=False, block_size=block_size, vocab_size=1
        )
        model = BaselineGPT(config)
        
    elif model_type == 'ddl':
        config = DDLConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            dropout=0, bias=False, block_size=block_size, vocab_size=1
        )
        model = DDLGPT(config)
        
    elif model_type == 'mhc':
        config = mHCConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_streams=n_streams, dropout=0, bias=False,
            block_size=block_size, vocab_size=1
        )
        model = mHCGPT(config)
        
    elif model_type == 'jpmhc':
        config = JPmHCConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_streams=n_streams, dropout=0, bias=False,
            block_size=block_size, vocab_size=1,
            cayley_alpha=0.1, cayley_iters=2
        )
        model = JPmHCGPT(config)

    elif model_type == 'edelta':
        config = EdeltaConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_streams=n_streams, dropout=0, bias=False,
            block_size=block_size, vocab_size=1,
            geo_hidden_ratio=4, use_mhc_projections=True
        )
        model = EdeltaGPT(config)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model


def compare_all_models(n_layer: int = 6, n_head: int = 4, n_embd: int = 128,
                       n_streams: int = 4, block_size: int = 64,
                       matched: bool = False) -> dict:
    """
    Compare parameter counts across all model architectures.
    
    When matched=True, uses per-model configs to match E∆-MHC-Geo's param count.
    JPmHC requires n_embd=512 (paper-faithful: F operates at d_stream width).
    
    Returns:
        dict: {model_name: param_count}
    """
    MATCH_CONFIG = {
        'gpt':    {'n_layer': 9},
        'ddl':    {'n_layer': 8},
        'mhc':    {'n_layer': 9},
        'jpmhc':  {'n_layer': 7, 'n_embd': 512},
        'edelta': {'n_layer': 6},
    }
    
    results = {}
    for model_type in ['gpt', 'ddl', 'mhc', 'jpmhc', 'edelta']:
        if matched:
            cfg = MATCH_CONFIG.get(model_type, {})
            ml = cfg.get('n_layer', n_layer)
            me = cfg.get('n_embd', n_embd)
        else:
            ml, me = n_layer, n_embd
        model = create_model(model_type, ml, n_head, me, n_streams, block_size)
        results[model_type] = count_params(model)
    
    return results


def find_matching_config(target_params: int, model_type: str, n_head: int = 4, 
                         n_embd: int = 128, n_streams: int = 4, 
                         block_size: int = 64, max_layers: int = 20) -> dict:
    """
    Find the config that gives closest parameter count to target.
    
    For JPmHC, also searches over n_embd since its sub-layer F operates at
    d_stream = n_embd // n_streams (paper Section 3.2), requiring wider n_embd
    to get sufficient per-layer capacity.
    
    Prefers practical depth (<=20 layers) over extreme depth.
    
    Returns:
        dict with keys: n_layer, n_embd, params, ratio
    """
    best = {'n_layer': None, 'n_embd': n_embd, 'params': None, 'ratio': 0}
    best_diff = float('inf')
    
    embd_candidates = [n_embd]
    if model_type == 'jpmhc':
        embd_candidates = [512, 256, 128]
    
    for me in embd_candidates:
        for nl in range(1, max_layers + 1):
            try:
                model = create_model(model_type, nl, n_head, me, n_streams, block_size)
                params = count_params(model)
                diff = abs(params - target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best = {'n_layer': nl, 'n_embd': me, 'params': params,
                            'ratio': params / target_params}
                if params > target_params * 1.5:
                    break
            except Exception:
                continue
        if best['params'] and abs(best['params'] - target_params) / target_params < 0.05:
            break
    
    return best


def find_matching_nlayer(target_params: int, model_type: str, n_head: int = 4, 
                         n_embd: int = 128, n_streams: int = 4, 
                         block_size: int = 64, max_layers: int = 100) -> tuple:
    """
    Find the n_layer value that gives closest parameter count to target.
    Backward-compatible wrapper around find_matching_config.
    
    Returns:
        (best_n_layer, best_params, ratio)
    """
    result = find_matching_config(target_params, model_type, n_head, n_embd,
                                  n_streams, block_size, max_layers)
    return result['n_layer'], result['params'], result['ratio']


def print_comparison_table(results: dict, reference: str = 'edelta'):
    """Print a formatted comparison table."""
    ref_params = results.get(reference, list(results.values())[0])
    
    print(f"\n{'Model':<12} {'Parameters':>12} {'Millions':>10} {'Ratio':>8}")
    print("-" * 44)
    
    for name, params in results.items():
        ratio = params / ref_params
        marker = " (ref)" if name == reference else ""
        print(f"{name:<12} {params:>12,} {params/1e6:>10.3f}M {ratio:>7.3f}x{marker}")


def print_component_breakdown(model: nn.Module, model_name: str):
    """Print parameter breakdown by component."""
    components = count_params_by_component(model)
    total = sum(components.values())
    
    print(f"\n{model_name} Component Breakdown:")
    print("-" * 50)
    
    for component, params in components.items():
        pct = 100 * params / total
        print(f"  {component:<25} {params:>10,} ({pct:5.1f}%)")
    
    print("-" * 50)
    print(f"  {'TOTAL':<25} {total:>10,}")


def main():
    parser = argparse.ArgumentParser(
        description='Parameter Counter Utility for E∆-MHC-Geo Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic comparison with default settings
    uv run src/utils/param_counter.py
    
    # Custom configuration
    uv run src/utils/param_counter.py --n_layer 6 --n_embd 128
    
    # Find n_layer values for baselines to match E∆-MHC-Geo
    uv run src/utils/param_counter.py --find_match
    
    # Show component breakdown for a specific model
    uv run src/utils/param_counter.py --breakdown edelta
        """
    )
    
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_streams', type=int, default=4, help='Number of streams (mHC/E∆)')
    parser.add_argument('--block_size', type=int, default=64, help='Block size')
    
    parser.add_argument('--find_match', action='store_true',
                        help='Find n_layer values for baselines to match E∆-MHC-Geo')
    parser.add_argument('--breakdown', type=str, choices=['gpt', 'ddl', 'mhc', 'jpmhc', 'edelta'],
                        help='Show component breakdown for specified model')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress model initialization messages')
    
    args = parser.parse_args()
    
    # Redirect stdout temporarily if quiet mode
    if args.quiet:
        import io
        import contextlib
        
        @contextlib.contextmanager
        def suppress_stdout():
            with contextlib.redirect_stdout(io.StringIO()):
                yield
    else:
        @contextlib.contextmanager
        def suppress_stdout():
            yield
    
    print("=" * 60)
    print("E∆-MHC-Geo Parameter Counter Utility")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  n_layer:    {args.n_layer}")
    print(f"  n_head:     {args.n_head}")
    print(f"  n_embd:     {args.n_embd}")
    print(f"  n_streams:  {args.n_streams}")
    print(f"  block_size: {args.block_size}")
    
    if args.find_match:
        # Find matching n_layer values for baselines
        print("\n" + "=" * 60)
        print("FINDING n_layer VALUES TO MATCH E∆-MHC-Geo")
        print("=" * 60)
        
        # First, get E∆-MHC-Geo reference params
        with suppress_stdout():
            edelta_model = create_model('edelta', args.n_layer, args.n_head, 
                                        args.n_embd, args.n_streams, args.block_size)
        target_params = count_params(edelta_model)
        print(f"\nReference: E∆-MHC-Geo (n_layer={args.n_layer})")
        print(f"Target: {target_params:,} ({target_params/1e6:.3f}M)")
        
        print(f"\n{'Model':<8} {'n_layer':>8} {'n_embd':>8} {'Parameters':>12} {'Millions':>10} {'Ratio':>8}")
        print("-" * 60)
        print(f"{'edelta':<8} {args.n_layer:>8} {args.n_embd:>8} {target_params:>12,} {target_params/1e6:>10.3f}M {'1.000x':>8}")
        
        for model_type in ['gpt', 'ddl', 'mhc', 'jpmhc']:
            with suppress_stdout():
                result = find_matching_config(
                    target_params, model_type, args.n_head, args.n_embd,
                    args.n_streams, args.block_size
                )
            nl, me = result['n_layer'], result['n_embd']
            params, ratio = result['params'], result['ratio']
            print(f"{model_type:<8} {nl:>8} {me:>8} {params:>12,} {params/1e6:>10.3f}M {ratio:>7.3f}x")
        
        print("\n" + "-" * 60)
        print("Recommended BASELINE_MATCH_CONFIG for --match_proposed_params:")
        print("BASELINE_MATCH_CONFIG = {")
        for model_type in ['gpt', 'ddl', 'mhc', 'jpmhc']:
            with suppress_stdout():
                result = find_matching_config(
                    target_params, model_type, args.n_head, args.n_embd,
                    args.n_streams, args.block_size
                )
            nl, me = result['n_layer'], result['n_embd']
            name = 'gpt2' if model_type == 'gpt' else model_type
            embd_str = f", 'n_embd': {me}" if me != args.n_embd else ""
            print(f"    '{name}': {{'n_layer': {nl}{embd_str}}},  # {result['params']/1e6:.3f}M")
        print("}")
        
    elif args.breakdown:
        # Show component breakdown for specified model
        print(f"\n" + "=" * 60)
        print(f"COMPONENT BREAKDOWN: {args.breakdown.upper()}")
        print("=" * 60)
        
        with suppress_stdout():
            model = create_model(args.breakdown, args.n_layer, args.n_head,
                                args.n_embd, args.n_streams, args.block_size)
        print_component_breakdown(model, args.breakdown.upper())
        
    else:
        # Basic comparison
        print("\n" + "=" * 60)
        print("PARAMETER COMPARISON")
        print("=" * 60)
        
        results = {}
        for model_type in ['gpt', 'ddl', 'mhc', 'jpmhc', 'edelta']:
            with suppress_stdout():
                model = create_model(model_type, args.n_layer, args.n_head,
                                    args.n_embd, args.n_streams, args.block_size)
            results[model_type] = count_params(model)
        
        print_comparison_table(results)
        
        # Show how much larger E∆ is
        edelta_params = results['edelta']
        gpt_params = results['gpt']
        overhead = edelta_params - gpt_params
        
        print(f"\nE∆-MHC-Geo overhead vs GPT: {overhead:,} ({100*overhead/gpt_params:.1f}%)")
        print("\nTo match parameters, use: --find_match")


if __name__ == '__main__':
    main()
