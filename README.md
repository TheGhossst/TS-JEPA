# TS-JEPA (paper-faithful baseline)

Reproduction of **Time-Series JEPA** for predictive remote control under capacity-limited networks. The implementation follows [`docs/plan.md`](docs/plan.md) and [`docs/TS-JEPA_Paper-Faithful_Final.md`](docs/TS-JEPA_Paper-Faithful_Final.md).

Choices that are not paper-specified are documented in [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md).

GE-JEPA is intentionally excluded until this baseline is validated.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

## Paper-scale baseline (recommended)

Use `configs/ts_jepa_dp_fixed.yaml` with the corrected DP-teacher dataset (`data_dp_fixed/`). Training writes to `runs/ts_jepa_dp_fixed/` and `runs/semantic_actor_dp_fixed/`.

| Stage | JEPA | Semantic actor |
|-------|------|----------------|
| Seeds | 5 (0–4) | 5 (0–4) |
| Epochs per seed | 150 | 300 |
| Selection | Best validation cosine loss | Best validation MSE |

Early stopping is **disabled** for JEPA — each seed runs the full 150 epochs. `best.pt` is still chosen by validation loss.

### One command — full pipeline

Generates data, trains JEPA (5×150), trains actor (5×300), and runs evaluation:

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Resume after interruption (skips completed seeds; re-runs only what is missing):

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate
```

### Step by step

**1. Validate environment (plan §3)**

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
```

**2. Generate trajectories (plan §4)**

```powershell
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
```

Produces 200/40 JEPA and 100/20 actor train/test trajectories under `data_dp_fixed/trajectories/`.

**3. Train TS-JEPA — 5 seeds × 150 epochs (plan §6–§13)**

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Outputs per seed under `runs/ts_jepa_dp_fixed/seed_{0..4}/` plus `runs/ts_jepa_dp_fixed/best.pt` and `repetition_summary.json`.

Train a single seed only:

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2
```

Resume an interrupted seed (per-epoch `last.pt` checkpoints):

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume-from runs/ts_jepa_dp_fixed/seed_2/last.pt
```

Or use the default checkpoint path for that seed:

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume
```

**4. Train semantic actor — 5 seeds × 300 epochs (plan §14+)**

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

**5. Evaluate (plan §16)**

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml
```

### Partial pipeline runs

```powershell
# Data already generated — train JEPA only
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate

# JEPA already trained — actor + eval only
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa

# Evaluation only
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa --skip-actor
```

Backward-compatible shortcuts still work (e.g. `python scripts/train_jepa.py`).

## Repository layout

```
TS-JEPA/
├── configs/                 # YAML configs (baseline + experiment overlays)
├── data_dp_fixed/           # Generated trajectories (JEPA + actor splits)
├── docs/                    # Plan, paper notes, implementation choices
├── runs/                    # Training checkpoints and evaluation outputs
├── scripts/
│   ├── pipeline/            # Main baseline entry points
│   ├── dev/                 # CUDA smoke / short-run helpers
│   ├── diagnose/            # Debugging and analysis
│   └── tools/               # Plotting utilities
├── src/ts_jepa/             # Installable Python package
│   ├── control/             # DP teacher controller
│   ├── data/                # Datasets, trajectory generation, temporal indexing
│   ├── env/                 # Custom inverted cart-pole RGB environment
│   ├── evaluation/          # Metrics, checkpoints, baseline report
│   ├── inference/           # Frozen runtime controller
│   ├── losses/              # JEPA cosine loss
│   ├── models/              # Encoder, predictor, actor, TS-JEPA module
│   ├── plan/                # Paper plan specs and config assertions (§3–§13)
│   ├── preprocessing/       # Augmentation, normalization, resize pipeline
│   ├── training/            # JEPA + actor training loops
│   └── wireless/            # Optional wireless eval (post-baseline)
└── tests/                   # Pytest suite
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/ts_jepa_baseline.yaml` | Paper-faithful hyperparameters and plan asserts |
| `configs/ts_jepa_dp_fixed.yaml` | Overlay: `data_dp_fixed/` paths and run directory names |

Key JEPA training settings (from baseline config):

- Optimizer: SGD, lr=0.2, weight decay=4e-4, effective batch=256, microbatch=16
- LR decay: ×0.99 every 20 epochs
- EMA target encoder decay: 0.99
- Early stopping: disabled (full 150 epochs per seed)

Plan assertions live in `src/ts_jepa/plan/` and are invoked from pipeline scripts before training.

## Development

```powershell
pytest
```

See [`scripts/README.md`](scripts/README.md) for dev, diagnose, and tool scripts.

## Key references

- [`docs/plan.md`](docs/plan.md) — section-by-section implementation plan
- [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md) — documented deviations from the paper
