# Commands

Run every command from the repository root (`C:\code\TS-JEPA`). Paths match script defaults and argparse.

GE-JEPA / burst channels are not included.

Training (JEPA, actor, probe decoder, supervised, autoencoder) is slow. Eval / diagnose commands below do not retrain unless marked **SLOW**.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Use `--device cuda` when a GPU is available.

---

## Observer-LQR (no training)

Requires existing checkpoints:

| Artifact | Default path |
|----------|----------------|
| MLP decoder | `runs/semantic_actor_working_probe_lqr_mlp/decoder.pt` |
| JEPA | `runs/ts_jepa_working/seed_0/best.pt` |
| Actor | `runs/semantic_actor_working/best.pt` |

### 5-seed Observer-LQR + ablations (seeds 100–104)

Writes `runs/semantic_actor_working_observer_lqr_5seed/metrics.json` (does not overwrite the default run dir).

JSON keys:

| Key | Meaning |
|-----|---------|
| `true_state_lqr` | Oracle full-state LQR |
| `true_position_observer_lqr_full_information` | Oracle `(x, θ)` + α-β observer + LQR |
| `observer_lqr_full_information` | Frozen-z decoded `(x, θ)` + α-β observer + LQR |
| `frozen_z_fd_lqr_full_information` | Frozen-z decoded `(x, θ)` + finite-difference velocities + LQR |
| `memoryless_probe_lqr_full_information` | Frozen-z memoryless Probe-LQR |
| `working_gate` | Receive-every-`Kp` (1/15) predict / hold-last / zero-action |

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3 --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_5seed
```

### Packet-loss sweep (Observer-LQR)

Same as above, plus Bernoulli receive rates `1.0, 0.8, 0.6, 0.4, 0.2` and periodic `1/15` (`Kp`). Compares **predict-only observer**, **hold-last**, and **zero-action** at each rate. Key: `packet_loss_sweep`.

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3 --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_packet_loss --packet-loss-sweep
```

Custom rates (token `1/15` is receive-every-Kp, not Bernoulli):

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_packet_loss --packet-receive-rates 1.0 0.8 0.6 0.4 0.2 1/15
```

Default output dir (overwrites `runs/semantic_actor_working_observer_lqr/metrics.json`), config seeds `[100, 101, 102]`:

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3
```

---

## Retrain the z→state decoder **SLOW**

Only if you need a new `decoder.pt`. Writes `runs/semantic_actor_working_probe_lqr_mlp/decoder.pt` and `metrics.json` (`probe_lqr_full_information`).

```powershell
python scripts/pipeline/train_actor_probe_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder mlp --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

Linear probe (writes `runs/semantic_actor_working_probe_lqr/probe_lqr.npz`):

```powershell
python scripts/pipeline/train_actor_probe_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder linear --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

---

## Simulation-only LQR / velocity diagnostics (no encoder)

True-state LQR noise / EMA sensitivity. Output: `runs/eval/lqr_noise_sensitivity.json` (look at `baseline_Q40_4_80_4_R0.05` → `noise=0.0_ema=1.0`).

```powershell
python scripts/diagnose/diagnose_lqr_noise_sensitivity.py --seeds 100 101 102 103 104 --out runs/eval/lqr_noise_sensitivity.json
```

Noisy-position velocity ablations (`true_vel`, `fd_vel`, `alpha_beta_vel`, attenuated-vel cases). Output: `runs/eval/velocity_observability_control.json`.

```powershell
python scripts/diagnose/diagnose_velocity_observability_control.py --seeds 100 101 102 103 104 --out runs/eval/velocity_observability_control.json
```

---

## Working overlay **SLOW** (not paper-faithful)

Config: `configs/ts_jepa_working.yaml`. Data: `data_working/`. Runs: `runs/ts_jepa_working/`, `runs/semantic_actor_working/`.

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda
```

Default is JEPA+actor seed 0. Full 5-seed protocol: `--all-seeds`.

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda --all-seeds
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda --skip-generate --skip-actor --jepa-epochs 20
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_working.yaml --device cuda --single-seed 0 --epochs 20 --resume-from runs/ts_jepa_working_contrast_broad_smoke/seed_0/last.pt
python scripts/pipeline/train_actor.py --config configs/ts_jepa_working.yaml --device cuda --single-seed 0 --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt
python scripts/pipeline/train_actor_dagger.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

DAgger writes `runs/semantic_actor_working_dagger/best.pt` and does not overwrite the BC actor.

Teacher-command range:

```powershell
python scripts/analysis/inspect_command_range.py --config configs/ts_jepa_working.yaml
```

---

## Paper pipeline **SLOW**

Config: `configs/ts_jepa_dp_fixed.yaml`. Data: `data_dp_fixed/`. Runs: `runs/ts_jepa_dp_fixed/`, `runs/semantic_actor_dp_fixed/`.

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Resume / partial:

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume-from runs/ts_jepa_dp_fixed/seed_2/last.pt
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa
```

---

## Runtime eval (FrozenRuntimeController / semantic actor)

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_working.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode nmae --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode tsne --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop_diagnostic --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --mode fig4 --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode wireless --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

Alternating-mask stability (semantic actor, not Observer-LQR): `runs/eval/stability_report.json` after `--mode baseline`.

---

## Paper experiments (Figs. 6–11)

**SLOW** trains:

```powershell
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 4
python scripts/pipeline/train_autoencoder.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig7 --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig8 --out-dir runs/eval/paper_experiments
```

Eval (Fig. 11 extra packet-loss rates are `[0.0, 0.1, 0.2, 0.3]` on FrozenRuntimeController / supervised, not Observer-LQR):

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment dp --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig6 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt --supervised-kappa4 runs/baselines/supervised_kappa4/seed_0/best.pt --autoencoder-checkpoint runs/baselines/autoencoder_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig9 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig11 --snr 10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
```

---

## Other diagnose

```powershell
python scripts/diagnose/trajectory_sanity_check.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/diagnose/diagnose_jepa_embeddings.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_raw_vs_embedding_collision.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_representation_pipeline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_dp_teacher.py
python scripts/diagnose/diagnose_actor_training.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_actor_predictions.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/probe_frozen_jepa_state.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt
python scripts/diagnose/probe_frozen_jepa_state_mlp.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
python scripts/diagnose/diagnose_raw_state_actor.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

---

## Tests

```powershell
pytest
pytest tests/test_observer_lqr.py tests/test_probe_lqr_actor.py
```


From a fresh clone, run everything in this order from `C:\code\TS-JEPA`. Use `--device cpu` if you have no GPU.

There are two tracks:

- **Track A (working overlay + Observer-LQR)** — what you need for the observer ablations
- **Track B (paper reproduction)** — full IEEE baseline + Figs. 6–11

---

## 0. Setup (once)

```powershell
cd C:\code\TS-JEPA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Optional sanity check:

```powershell
pytest
```

---

## Track A — Working overlay + Observer-LQR (from scratch)

### A1. Train everything (data → JEPA → actor → gates)

**SLOW** — hours on GPU. Seed 0 only by default:

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda
```

Full 5-seed JEPA + actor protocol:

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda --all-seeds
```

Or step by step:

```powershell
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_working.yaml
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_working.yaml --device cuda --single-seed 0
python scripts/pipeline/train_actor.py --config configs/ts_jepa_working.yaml --device cuda --single-seed 0 --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt
```

### A2. Train z→state MLP decoder (required for Observer-LQR)

**SLOW** — ~minutes:

```powershell
python scripts/pipeline/train_actor_probe_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder mlp --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

Writes `runs/semantic_actor_working_probe_lqr_mlp/decoder.pt`.

### A3. Observer-LQR eval (all ablations, 5 seeds)

Fast — no training:

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3 --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_5seed
```

Output: `runs/semantic_actor_working_observer_lqr_5seed/metrics.json`

### A4. Observer packet-loss sweep

```powershell
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3 --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_packet_loss --packet-loss-sweep
```

Output: `runs/semantic_actor_working_observer_lqr_packet_loss/metrics.json` → key `packet_loss_sweep`

### A5. Simulation-only LQR / velocity diagnostics (no encoder)

```powershell
python scripts/diagnose/diagnose_lqr_noise_sensitivity.py --seeds 100 101 102 103 104 --out runs/eval/lqr_noise_sensitivity.json
python scripts/diagnose/diagnose_velocity_observability_control.py --seeds 100 101 102 103 104 --out runs/eval/velocity_observability_control.json
```

### A6. Semantic-actor runtime eval (working overlay)

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_working.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --out-dir runs/eval/working
```

### A7. Optional working-overlay extras

```powershell
python scripts/analysis/inspect_command_range.py --config configs/ts_jepa_working.yaml
python scripts/diagnose/probe_frozen_jepa_state.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt
python scripts/diagnose/probe_frozen_jepa_state_mlp.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
python scripts/diagnose/diagnose_raw_state_actor.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
python scripts/pipeline/train_actor_dagger.py --config configs/ts_jepa_working.yaml --device cuda --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
```

---

## Track B — Paper reproduction (from scratch)

**SLOW** — days on GPU (5×150 JEPA + 5×300 actor epochs).

### B1. One-shot full pipeline

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

### B2. Or step by step

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/diagnose/trajectory_sanity_check.py --config configs/ts_jepa_dp_fixed.yaml
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

### B3. Train baselines for Figs. 6–11

```powershell
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 4
python scripts/pipeline/train_autoencoder.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig7 --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig8 --out-dir runs/eval/paper_experiments
```

### B4. Run paper experiments

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment dp --out-dir runs/eval/paper_experiments
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig6 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt --supervised-kappa4 runs/baselines/supervised_kappa4/seed_0/best.pt --autoencoder-checkpoint runs/baselines/autoencoder_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig9 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig11 --snr 10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
```

### B5. Extra runtime eval modes

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode nmae --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode tsne --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop_diagnostic --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --mode fig4 --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode wireless --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

### B6. Paper diagnostics

```powershell
python scripts/diagnose/diagnose_jepa_embeddings.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_raw_vs_embedding_collision.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_jepa_representation_pipeline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_dp_teacher.py
python scripts/diagnose/diagnose_actor_training.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
python scripts/diagnose/diagnose_actor_predictions.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

---

## Minimal “just Observer-LQR from scratch”

If you only care about the observer experiments and want the shortest path:

```powershell
cd C:\code\TS-JEPA
.\.venv\Scripts\Activate.ps1
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda
python scripts/pipeline/train_actor_probe_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder mlp --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt
python scripts/pipeline/eval_observer_lqr.py --config configs/ts_jepa_working.yaml --device cuda --decoder-checkpoint runs/semantic_actor_working_probe_lqr_mlp/decoder.pt --jepa-checkpoint runs/ts_jepa_working/seed_0/best.pt --actor-checkpoint runs/semantic_actor_working/best.pt --alpha 0.5 --beta 0.3 --seeds 100 101 102 103 104 --out-dir runs/semantic_actor_working_observer_lqr_5seed --packet-loss-sweep
python scripts/diagnose/diagnose_lqr_noise_sensitivity.py --seeds 100 101 102 103 104 --out runs/eval/lqr_noise_sensitivity.json
python scripts/diagnose/diagnose_velocity_observability_control.py --seeds 100 101 102 103 104 --out runs/eval/velocity_observability_control.json
```

---

## Expected outputs

| Step | Artifact |
|------|----------|
| Working train | `runs/ts_jepa_working/`, `runs/semantic_actor_working/` |
| Probe decoder | `runs/semantic_actor_working_probe_lqr_mlp/decoder.pt` |
| Observer eval | `runs/semantic_actor_working_observer_lqr_5seed/metrics.json` |
| Packet-loss sweep | `runs/semantic_actor_working_observer_lqr_packet_loss/metrics.json` |
| LQR noise sim | `runs/eval/lqr_noise_sensitivity.json` |
| Velocity sim | `runs/eval/velocity_observability_control.json` |
| Paper baseline | `runs/ts_jepa_dp_fixed/`, `runs/semantic_actor_dp_fixed/`, `runs/eval/` |
