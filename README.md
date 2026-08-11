# TS-JEPA (paper-faithful baseline)

Reproduction of **Time-Series JEPA** for predictive remote control under capacity-limited networks. The implementation follows [`docs/plan.md`](docs/plan.md) and [`docs/TS-JEPA_Paper-Faithful_Final.md`](docs/TS-JEPA_Paper-Faithful_Final.md).

Choices that are not paper-specified are documented in [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md).

GE-JEPA is intentionally excluded until this baseline is validated.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

## Quick start

Recommended config for the corrected DP-teacher dataset (`data_dp_fixed/`):

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
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

## Pipeline

### 1. Validate environment (plan §3)

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
```

### 2. Generate trajectories (plan §4)

```powershell
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
```

Produces 200/40 JEPA and 100/20 actor train/test trajectories under `data_dp_fixed/trajectories/`.

### 3. Train TS-JEPA (plan §6–§13)

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

5-seed protocol (150 epochs, SGD lr=0.2, batch 256). Use `--single-seed N` for one seed.

### 4. Train semantic actor (plan §14+)

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

### 5. Evaluate

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/ts_jepa_baseline.yaml` | Paper-faithful hyperparameters and plan asserts |
| `configs/ts_jepa_dp_fixed.yaml` | Overlay: `data_dp_fixed/` paths and run directory names |

Plan assertions live in `src/ts_jepa/plan/` and are invoked from pipeline scripts before training.

## Development

```powershell
pytest
```

See [`scripts/README.md`](scripts/README.md) for dev, diagnose, and tool scripts.

## Key references

- [`docs/plan.md`](docs/plan.md) — section-by-section implementation plan
- [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md) — documented deviations from the paper
