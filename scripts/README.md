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

## Full paper-scale pipeline

**Config:** `configs/ts_jepa_dp_fixed.yaml`  
**Data:** `data_dp_fixed/`  
**Runs:** `runs/ts_jepa_dp_fixed/`, `runs/semantic_actor_dp_fixed/`

### One command

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Runs: generate trajectories → JEPA 5×150 epochs → actor 5×300 epochs → evaluation.

### Step by step

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml
```

### JEPA training (5 seeds × 150 epochs)

Full 5-seed protocol (selects best validation run → `runs/ts_jepa_dp_fixed/best.pt`):

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Single seed:

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2
```

Resume interrupted seed (uses per-epoch `last.pt`):

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume-from runs/ts_jepa_dp_fixed/seed_2/last.pt
```

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume
```

The 5-seed protocol skips seeds that already finished (final `last.pt` with `test_loss`). Completed seeds are not retrained unless you remove their run directory.

### Semantic actor (SAM) — debug on JEPA seed 0, then final 5-seed training

**Debug now** (JEPA seed 0 encoder, while JEPA seeds 1–4 continue):

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 0 --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode nmae --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/seed_0/best.pt --out-dir runs/eval/sam_debug_seed0
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/seed_0/best.pt --out-dir runs/eval/sam_debug_seed0
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode tsne --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/seed_0/best.pt --out-dir runs/eval/sam_debug_seed0
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --mode fig4 --out-dir runs/eval
```

**Final SAM** after all 5 JEPA seeds (uses selected `runs/ts_jepa_dp_fixed/best.pt`):

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt
```

**Baseline validation (plan §15) before wireless:**

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

### Partial / resume pipeline

```powershell
# Skip data generation (data already on disk)
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate

# JEPA already done — actor + eval only
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa

# Override epoch count (smoke / debug only)
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --jepa-epochs 3
```

## Plan §16–§17 baselines / Figs. 6–11

```powershell
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 4
python scripts/pipeline/train_autoencoder.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment dp --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig6 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt --supervised-kappa4 runs/baselines/supervised_kappa4/seed_0/best.pt --autoencoder-checkpoint runs/baselines/autoencoder_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig7 --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig8 --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig9 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig11 --snr 10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
```

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
| `diagnose/trajectory_sanity_check.py` | Verify on-disk trajectory dataset (counts, frames, IDs, commands) |
| `diagnose/diagnose_jepa_embeddings.py` | Embedding statistics and collision checks |
| `diagnose/diagnose_jepa_raw_vs_embedding_collision.py` | Raw vs latent collision analysis |
| `diagnose/diagnose_jepa_representation_pipeline.py` | Full representation pipeline audit |
| `diagnose/diagnose_dp_teacher.py` | DP teacher control inspection |
| `diagnose/diagnose_actor_training.py` | Actor training diagnostics |
| `diagnose/diagnose_actor_predictions.py` | Actor prediction analysis |

```powershell
# 2. Verify dataset
python scripts/diagnose/trajectory_sanity_check.py --config configs/ts_jepa_dp_fixed.yaml

python scripts/diagnose/diagnose_jepa_embeddings.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_raw_vs_embedding_collision.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_representation_pipeline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_dp_teacher.py
python scripts/diagnose/diagnose_actor_training.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_actor_predictions.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

## Tools

| Script | Description |
|--------|-------------|
| `tools/plot_eval.py` | Plot evaluation artifacts |
