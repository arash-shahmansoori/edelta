# ARC-Reflect-Lite Protocol

This protocol isolates the specific claim that reflection capability adds value over JPmHC on ARC tasks, while keeping GPU cost low.

## 1) Hypothesis and Design

- **Hypothesis**: `edelta` should outperform `jpmhc` mainly on reflection-sensitive tasks, with a smaller (or no) gap on rotation/control tasks.
- **Backbone control**: Use the same TRM backbone and training setup; only `arch.mixer_type` changes (`none`, `jpmhc`, `edelta`).
- **Task slices**:
  - `reflect`: tasks where train pairs are better explained by reflection-like transforms.
  - `control`: tasks where train pairs are better explained by non-reflection transforms (identity/rotation) with low reflection fit.
- **Primary metric**: `Pass@1` on each slice.
- **Primary effect**: Reflection Gain Index (RGI):
  - `RGI(model) = Pass@1_reflect(model) - Pass@1_control(model)`
  - `Delta_RGI = RGI(edelta) - RGI(jpmhc)`

## 2) Build ARC-Reflect-Lite Slices

Run from `arc_agi_trm/`:

```bash
python -m dataset.build_arc_reflect_lite \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-prefix data/arc_reflect_lite/arc-agi \
  --subsets training evaluation concept \
  --eval-subset-name evaluation \
  --reflect-min-fit 0.3 \
  --reflect-min-margin 0.0 \
  --control-min-non-reflect-fit 0.2 \
  --control-max-reflect-fit 0.2 \
  --control-min-margin 0.0 \
  --max-tasks-per-group 20 \
  --min-eval-tasks-per-group 5 \
  --seed 42
```

With current ARC-AGI-1 files and `seed=42`, a dry run yields:

- reflect candidates: 20
- control candidates: 34
- selected: 20 reflect + 20 control

What this creates:

- `data/arc_reflect_lite/arc-agi_manifest.json` (full scoring + selected IDs)
- `data/arc_reflect_lite/arc-agi_reflect_{subset}_{challenges,solutions}.json`
- `data/arc_reflect_lite/arc-agi_control_{subset}_{challenges,solutions}.json`

## 3) Convert Filtered JSON to Training Datasets

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix data/arc_reflect_lite/arc-agi_reflect \
  --output-dir data/arc_reflect_lite_reflect \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 128
```

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix data/arc_reflect_lite/arc-agi_control \
  --output-dir data/arc_reflect_lite_control \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 128
```

## 4) Low-Cost Training Runs

Use the provided runner:

```bash
bash scripts/run_arc_reflect_lite.sh all
```

Default run settings are intentionally lightweight (`EPOCHS=80`, `GLOBAL_BATCH_SIZE=128`).  
Override for very small pilot:

```bash
EPOCHS=30 EVAL_INTERVAL=5 GLOBAL_BATCH_SIZE=64 bash scripts/run_arc_reflect_lite.sh all
```

## 5) Reporting Template (Paper-Ready)

Report a table like:

| Mixer | Pass@1 Reflect | Pass@1 Control | RGI |
|---|---:|---:|---:|
| none | ... | ... | ... |
| jpmhc | ... | ... | ... |
| edelta | ... | ... | ... |

Then report:

- `Delta_reflect = Pass@1_reflect(edelta) - Pass@1_reflect(jpmhc)`
- `Delta_control = Pass@1_control(edelta) - Pass@1_control(jpmhc)`
- `Delta_RGI = (Delta_reflect - Delta_control)`

Interpretation:

- If `Delta_reflect > 0` and `Delta_RGI > 0`, this supports the specific added value of reflection.
- If `Delta_control` is near zero, the gain is targeted rather than a generic capacity gain.

## 6) Significance and Credibility Checks

- Run at least `3` seeds (`42, 43, 44`) for final numbers.
- Use paired bootstrap over evaluation tasks for `Delta_reflect` and `Delta_RGI` (10k resamples).
- Publish the manifest (`arc-agi_manifest.json`) in the paper repo so task selection is auditable.
- Keep selector thresholds fixed before final runs (`reflect_min_*`, `control_*`, `max-tasks-per-group`).

## 7) Suggested Compute Budget

- **Pilot**: 1 seed, all three mixers, 30 epochs.
- **Final lite study**: 3 seeds, `jpmhc` + `edelta` (plus optional `none`), 80 epochs.
- This is substantially cheaper than full ARC-AGI sweeps while still testing the core mechanistic claim.
