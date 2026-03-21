# ARC Reflect Selector Spec

This is the minimal spec implemented by `dataset/build_arc_reflect_lite.py`.

## Inputs

- Raw ARC files:
  - `<input_prefix>_<subset>_challenges.json`
  - `<input_prefix>_<subset>_solutions.json` (optional)
- Subsets list (default: `training evaluation concept`)
- Thresholds:
  - `reflect_min_fit` (default `0.3`, via `--reflect-min-fit` or legacy `--min-fit`)
  - `reflect_min_margin` (default `0.0`, via `--reflect-min-margin` or legacy `--min-margin`)
  - `control_min_non_reflect_fit` (default `0.2`)
  - `control_max_reflect_fit` (default `0.2`)
  - `control_min_margin` (default `0.0`)
  - `max_tasks_per_group` (default `20`)
  - `min_eval_tasks_per_group` (default `5`)

## Per-Puzzle Scoring

For each train pair `(input, output)`:

1. Test all 8 dihedral transforms `tid in [0..7]` on `input`.
2. For each `tid`, accept if there exists a **consistent color remap** from transformed input colors to output colors.
3. Record pair-level hits:
   - reflection hit if any `tid in {4,5,6,7}`
   - rotation hit if any `tid in {1,2,3}`
   - identity hit if `tid == 0`
   - non-reflect hit if any `tid in {0,1,2,3}`

Aggregate by puzzle:

- `reflect_fit = reflection_hits / num_train_pairs`
- `rotation_fit = rotation_hits / num_train_pairs`
- `identity_fit = identity_hits / num_train_pairs`
- `non_reflect_fit = non_reflect_hits / num_train_pairs`
- `reflect_margin = reflect_fit - max(rotation_fit, identity_fit)`
- `control_margin = max(rotation_fit, identity_fit) - reflect_fit`

## Labeling Rule

- `reflect` if:
  - `reflect_fit >= reflect_min_fit`, and
  - `reflect_margin >= reflect_min_margin`
- `control` if:
  - `max(rotation_fit, identity_fit) >= control_min_non_reflect_fit`, and
  - `reflect_fit <= control_max_reflect_fit`, and
  - `control_margin >= control_min_margin`
- else `other`

## Selection Rule

- Build candidate pools for `reflect` and `control`.
- If pool size exceeds `max_tasks_per_group`, sample with fixed seed.
- Try to include at least `min_eval_tasks_per_group` from the eval subset (`evaluation` by default), then fill remaining slots randomly.
- This naturally yields balanced final slices when both pools exceed `max_tasks_per_group`.

## Outputs

Using `output_prefix`:

- Filtered JSONs:
  - `<output_prefix>_reflect_<subset>_challenges.json`
  - `<output_prefix>_reflect_<subset>_solutions.json`
  - `<output_prefix>_control_<subset>_challenges.json`
  - `<output_prefix>_control_<subset>_solutions.json`
- Manifest:
  - `<output_prefix>_manifest.json` with config, effective thresholds, counts, selected IDs, and per-puzzle records.

## Reproducibility

- All random choices are controlled by `--seed`.
- Manifest is the audit artifact for exact task composition.
