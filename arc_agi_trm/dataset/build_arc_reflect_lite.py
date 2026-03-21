#!/usr/bin/env python3
"""
Build reflection-sensitive and control ARC subsets for low-cost benchmarking.

This script scores each puzzle from ARC challenge JSON files using train pairs:
- reflection fit: fraction of pairs solvable by any reflection-like dihedral map
- rotation fit: fraction of pairs solvable by any proper rotation-like dihedral map
- identity fit: fraction of pairs solvable by identity map
- non-reflect fit: fraction of pairs solvable by identity/rotation maps

The script then selects two puzzle groups:
- reflect: reflection-dominant tasks
- control: non-reflection-dominant tasks (identity/rotation, low reflection)

Output:
- filtered challenge/solution JSON files per subset and group
- a manifest with per-puzzle scores and selected IDs
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


ROTATION_TIDS = (1, 2, 3)
REFLECTION_TIDS = (4, 5, 6, 7)
NON_REFLECTION_TIDS = (0, 1, 2, 3)


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror."""
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


@dataclass
class PuzzleRecord:
    subset: str
    puzzle_id: str
    reflect_fit: float
    rotation_fit: float
    identity_fit: float
    non_reflect_fit: float
    num_train_pairs: int
    margin_reflect_minus_rotation: float
    margin_reflect_minus_non_reflect: float
    label: str


def _load_subset(input_prefix: str, subset: str) -> Tuple[Dict, Dict]:
    challenges_path = f"{input_prefix}_{subset}_challenges.json"
    solutions_path = f"{input_prefix}_{subset}_solutions.json"

    with open(challenges_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    solutions = {}
    if os.path.isfile(solutions_path):
        with open(solutions_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

    return challenges, solutions


def _fits_with_color_remap(inp: np.ndarray, out: np.ndarray, tid: int) -> bool:
    transformed = dihedral_transform(inp, tid)
    if transformed.shape != out.shape:
        return False

    # ARC colors are in [0..9], so we can build a compact LUT.
    # lut[c] == mapped output color for input color c.
    lut = np.full(10, -1, dtype=np.int16)

    in_flat = transformed.reshape(-1)
    out_flat = out.reshape(-1)
    for in_color, out_color in zip(in_flat, out_flat):
        in_color = int(in_color)
        out_color = int(out_color)
        prev = lut[in_color]
        if prev == -1:
            lut[in_color] = out_color
        elif prev != out_color:
            return False

    mapped = lut[transformed]
    return np.array_equal(mapped, out)


def _score_puzzle(puzzle: Dict) -> Tuple[float, float, float, float, int]:
    train_pairs = puzzle.get("train", [])
    if not train_pairs:
        return 0.0, 0.0, 0.0, 0.0, 0

    reflection_hits = 0
    rotation_hits = 0
    identity_hits = 0
    non_reflect_hits = 0

    for pair in train_pairs:
        inp = np.array(pair["input"], dtype=np.uint8)
        out = np.array(pair["output"], dtype=np.uint8)

        hits = []
        for tid in range(8):
            if _fits_with_color_remap(inp, out, tid):
                hits.append(tid)

        if any(tid in REFLECTION_TIDS for tid in hits):
            reflection_hits += 1
        if any(tid in ROTATION_TIDS for tid in hits):
            rotation_hits += 1
        if 0 in hits:
            identity_hits += 1
        if any(tid in NON_REFLECTION_TIDS for tid in hits):
            non_reflect_hits += 1

    n_pairs = len(train_pairs)
    return (
        reflection_hits / n_pairs,
        rotation_hits / n_pairs,
        identity_hits / n_pairs,
        non_reflect_hits / n_pairs,
        n_pairs,
    )


def _label_puzzle(
    reflect_fit: float,
    rotation_fit: float,
    identity_fit: float,
    reflect_min_fit: float,
    reflect_min_margin: float,
    control_min_non_reflect_fit: float,
    control_max_reflect_fit: float,
    control_min_margin: float,
) -> str:
    non_reflect_fit = max(rotation_fit, identity_fit)
    reflect_margin = reflect_fit - non_reflect_fit
    control_margin = non_reflect_fit - reflect_fit

    if reflect_fit >= reflect_min_fit and reflect_margin >= reflect_min_margin:
        return "reflect"
    if (
        non_reflect_fit >= control_min_non_reflect_fit
        and reflect_fit <= control_max_reflect_fit
        and control_margin >= control_min_margin
    ):
        return "control"
    return "other"


def _random_pick_with_eval_floor(
    records: List[PuzzleRecord],
    max_tasks: int,
    min_eval_tasks: int,
    eval_subset: str,
    rng: np.random.Generator,
) -> List[PuzzleRecord]:
    if max_tasks <= 0 or len(records) <= max_tasks:
        return list(records)

    eval_records = [r for r in records if r.subset == eval_subset]
    non_eval_records = [r for r in records if r.subset != eval_subset]

    chosen: List[PuzzleRecord] = []
    if min_eval_tasks > 0 and eval_records:
        eval_idx = rng.choice(
            len(eval_records),
            size=min(min_eval_tasks, len(eval_records), max_tasks),
            replace=False,
        )
        chosen.extend(eval_records[i] for i in eval_idx.tolist())

    remaining_slots = max_tasks - len(chosen)
    if remaining_slots <= 0:
        return chosen

    chosen_keys = {(r.subset, r.puzzle_id) for r in chosen}
    pool = [r for r in (eval_records + non_eval_records) if (r.subset, r.puzzle_id) not in chosen_keys]

    if not pool:
        return chosen

    pool_idx = rng.choice(len(pool), size=min(remaining_slots, len(pool)), replace=False)
    chosen.extend(pool[i] for i in pool_idx.tolist())
    return chosen


def _write_group_files(
    output_prefix: str,
    group_name: str,
    subsets: List[str],
    selected: List[PuzzleRecord],
    raw_challenges: Dict[str, Dict],
    raw_solutions: Dict[str, Dict],
) -> None:
    by_subset = {subset: set() for subset in subsets}
    for rec in selected:
        by_subset[rec.subset].add(rec.puzzle_id)

    for subset in subsets:
        ids = by_subset[subset]
        challenges = raw_challenges.get(subset, {})
        solutions = raw_solutions.get(subset, {})

        filtered_challenges = {pid: challenges[pid] for pid in ids if pid in challenges}
        filtered_solutions = {pid: solutions[pid] for pid in ids if pid in solutions}

        challenge_out = f"{output_prefix}_{group_name}_{subset}_challenges.json"
        solution_out = f"{output_prefix}_{group_name}_{subset}_solutions.json"

        with open(challenge_out, "w", encoding="utf-8") as f:
            json.dump(filtered_challenges, f)
        with open(solution_out, "w", encoding="utf-8") as f:
            json.dump(filtered_solutions, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ARC Reflect-Lite puzzle subsets.")
    parser.add_argument(
        "--input-file-prefix",
        type=str,
        required=True,
        help="Raw ARC prefix, e.g. kaggle/combined/arc-agi",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix for filtered JSON files, e.g. data/arc_reflect_lite/arc-agi",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["training", "evaluation", "concept"],
        help="Subsets to scan and filter.",
    )
    parser.add_argument(
        "--eval-subset-name",
        type=str,
        default="evaluation",
        help="Subset considered evaluation for minimum eval-task enforcement.",
    )
    parser.add_argument(
        "--min-fit",
        type=float,
        default=0.3,
        help="Legacy fallback for reflect minimum fit (use --reflect-min-fit).",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Legacy fallback for reflect minimum margin (use --reflect-min-margin).",
    )
    parser.add_argument(
        "--reflect-min-fit",
        type=float,
        default=None,
        help="Minimum reflection fit for reflect slice.",
    )
    parser.add_argument(
        "--reflect-min-margin",
        type=float,
        default=None,
        help="Minimum (reflect_fit - max(rotation_fit, identity_fit)) for reflect slice.",
    )
    parser.add_argument(
        "--control-min-non-reflect-fit",
        type=float,
        default=0.2,
        help="Minimum max(rotation_fit, identity_fit) for control slice.",
    )
    parser.add_argument(
        "--control-max-reflect-fit",
        type=float,
        default=0.2,
        help="Maximum reflection fit allowed for control slice.",
    )
    parser.add_argument(
        "--control-min-margin",
        type=float,
        default=0.0,
        help="Minimum (max(rotation_fit, identity_fit) - reflect_fit) for control slice.",
    )
    parser.add_argument(
        "--max-tasks-per-group",
        type=int,
        default=20,
        help="Maximum selected tasks for each group. <=0 means no cap.",
    )
    parser.add_argument(
        "--min-eval-tasks-per-group",
        type=int,
        default=5,
        help="Try to keep at least this many tasks from eval subset in each group.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    reflect_min_fit = args.reflect_min_fit if args.reflect_min_fit is not None else args.min_fit
    reflect_min_margin = args.reflect_min_margin if args.reflect_min_margin is not None else args.min_margin

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    raw_challenges: Dict[str, Dict] = {}
    raw_solutions: Dict[str, Dict] = {}
    all_records: List[PuzzleRecord] = []

    for subset in args.subsets:
        challenges, solutions = _load_subset(args.input_file_prefix, subset)
        raw_challenges[subset] = challenges
        raw_solutions[subset] = solutions

        for puzzle_id, puzzle in challenges.items():
            reflect_fit, rotation_fit, identity_fit, non_reflect_fit, n_pairs = _score_puzzle(puzzle)
            margin = reflect_fit - rotation_fit
            margin_non_reflect = reflect_fit - non_reflect_fit
            label = _label_puzzle(
                reflect_fit=reflect_fit,
                rotation_fit=rotation_fit,
                identity_fit=identity_fit,
                reflect_min_fit=reflect_min_fit,
                reflect_min_margin=reflect_min_margin,
                control_min_non_reflect_fit=args.control_min_non_reflect_fit,
                control_max_reflect_fit=args.control_max_reflect_fit,
                control_min_margin=args.control_min_margin,
            )
            all_records.append(
                PuzzleRecord(
                    subset=subset,
                    puzzle_id=puzzle_id,
                    reflect_fit=reflect_fit,
                    rotation_fit=rotation_fit,
                    identity_fit=identity_fit,
                    non_reflect_fit=non_reflect_fit,
                    num_train_pairs=n_pairs,
                    margin_reflect_minus_rotation=margin,
                    margin_reflect_minus_non_reflect=margin_non_reflect,
                    label=label,
                )
            )

    reflect_candidates = [r for r in all_records if r.label == "reflect"]
    control_candidates = [r for r in all_records if r.label == "control"]
    other_candidates = [r for r in all_records if r.label == "other"]

    rng = np.random.default_rng(args.seed)
    reflect_selected = _random_pick_with_eval_floor(
        records=reflect_candidates,
        max_tasks=args.max_tasks_per_group,
        min_eval_tasks=args.min_eval_tasks_per_group,
        eval_subset=args.eval_subset_name,
        rng=rng,
    )
    control_selected = _random_pick_with_eval_floor(
        records=control_candidates,
        max_tasks=args.max_tasks_per_group,
        min_eval_tasks=args.min_eval_tasks_per_group,
        eval_subset=args.eval_subset_name,
        rng=rng,
    )

    _write_group_files(
        output_prefix=args.output_prefix,
        group_name="reflect",
        subsets=args.subsets,
        selected=reflect_selected,
        raw_challenges=raw_challenges,
        raw_solutions=raw_solutions,
    )
    _write_group_files(
        output_prefix=args.output_prefix,
        group_name="control",
        subsets=args.subsets,
        selected=control_selected,
        raw_challenges=raw_challenges,
        raw_solutions=raw_solutions,
    )

    manifest = {
        "config": vars(args),
        "effective_thresholds": {
            "reflect_min_fit": reflect_min_fit,
            "reflect_min_margin": reflect_min_margin,
            "control_min_non_reflect_fit": args.control_min_non_reflect_fit,
            "control_max_reflect_fit": args.control_max_reflect_fit,
            "control_min_margin": args.control_min_margin,
        },
        "counts": {
            "total_scored": len(all_records),
            "reflect_candidates": len(reflect_candidates),
            "control_candidates": len(control_candidates),
            "other_candidates": len(other_candidates),
            "reflect_selected": len(reflect_selected),
            "control_selected": len(control_selected),
        },
        "selected_ids": {
            "reflect": [f"{r.subset}/{r.puzzle_id}" for r in reflect_selected],
            "control": [f"{r.subset}/{r.puzzle_id}" for r in control_selected],
        },
        "records": [asdict(r) for r in all_records],
    }

    manifest_path = f"{args.output_prefix}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("ARC Reflect-Lite selection complete")
    print(f"  Manifest: {manifest_path}")
    print(f"  Reflect candidates: {len(reflect_candidates)}")
    print(f"  Control candidates: {len(control_candidates)}")
    print(f"  Reflect selected: {len(reflect_selected)}")
    print(f"  Control selected: {len(control_selected)}")
    print(f"  Filtered prefix: {args.output_prefix}_{{reflect,control}}_*")


if __name__ == "__main__":
    main()
