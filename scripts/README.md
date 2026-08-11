# Scripts

CLI entry points for the TS-JEPA baseline. Run all commands from the repository root.

## Layout

| Folder | Purpose |
|--------|---------|
| `pipeline/` | End-to-end baseline: data generation, training, evaluation |
| `dev/` | CUDA smoke tests and short training runs |
| `diagnose/` | Debugging and representation analysis |
| `tools/` | Plotting and auxiliary utilities |

Root-level files such as `scripts/train_jepa.py` are thin wrappers that forward to `pipeline/` for backward compatibility.

## Pipeline (recommended)

Full paper-scale run with the corrected DP-teacher dataset:

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Step by step:

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml
```

Use `--single-seed N` on the train scripts to run one seed only.

## Dev scripts

| Script | Description |
|--------|-------------|
| `dev/cuda_smoke_jepa.py` | Minimal forward/backward smoke test on GPU |
| `dev/bench_jepa_accum_cuda.py` | Microbatch accumulation benchmark |
| `dev/one_epoch_jepa_cuda.py` | Single-epoch CUDA training check |
| `dev/validate_jepa_3epoch_cuda.py` | Short 3-epoch validation run |

## Diagnose scripts

| Script | Description |
|--------|-------------|
| `diagnose/diagnose_jepa_embeddings.py` | Embedding statistics and collision checks |
| `diagnose/diagnose_jepa_raw_vs_embedding_collision.py` | Raw vs latent collision analysis |
| `diagnose/diagnose_jepa_representation_pipeline.py` | Full representation pipeline audit |
| `diagnose/diagnose_dp_teacher.py` | DP teacher control inspection |
| `diagnose/diagnose_actor_training.py` | Actor training diagnostics |
| `diagnose/diagnose_actor_predictions.py` | Actor prediction analysis |

## Tools

| Script | Description |
|--------|-------------|
| `tools/plot_eval.py` | Plot evaluation artifacts |
