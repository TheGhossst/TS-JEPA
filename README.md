# TS-JEPA (paper-faithful baseline)

Reproduction of **Time-Series JEPA** for predictive remote control under capacity-limited networks.

**Source of truth:** [`docs/plan.md`](docs/plan.md). Executable gaps belong in [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md). Do not treat implementation-choice (IC) values as paper facts.

GE-JEPA / burst channels are excluded; they are not in the paper.

The published 1 ms + cosine-only recipe identity-collapses the predictor. For a **working** (non-paper) controller, use [`configs/ts_jepa_working.yaml`](configs/ts_jepa_working.yaml) and [`docs/WORKING.md`](docs/WORKING.md):

```powershell
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda
```

Run every command below from the **repository root** after a fresh clone.

## Environment and setup

Requires **Python ≥ 3.11**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Optional: `pip install -e ".[dev]"` instead of (or in addition to) `requirements.txt` for pytest.

Use `--device cuda` when a GPU is available, or `--device cpu` otherwise.

**Config:** `configs/ts_jepa_dp_fixed.yaml` (inherits `configs/ts_jepa_baseline.yaml`).  
**Data:** `data_dp_fixed/`  
**Checkpoints:** `runs/ts_jepa_dp_fixed/`, `runs/semantic_actor_dp_fixed/`  
**Eval artifacts:** `runs/eval/`

Paper protocol: **5 experimental repeats, report the best** (`evaluation.reported_result: best`). JEPA and actor **early stopping are enabled** (patience / val-split sizes are IC).

Regenerate trajectories if you have older `data_dp_fixed/` files stored at **64×128** or with overlapping D_s/D_a indices. Paper setup is \(\tau_o = 1\) ms, 100 steps. Native stored RGB is the IC camera **128×256**; encoder input is 64×128 after Gaussian resize. D_s uses trajectory indices 0–239 and D_a uses 240–359 (disjoint physical rollouts; paper-silent).

## One command — full pipeline

Generates data, trains TS-JEPA (5 seeds × 150 epochs), trains the semantic actor (5 seeds × 300 epochs), then runs plan §15 evaluation:

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Resume after interruption (skips completed seeds; re-runs only what is missing):

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate
```

Partial runs:

```powershell
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa --skip-actor
python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --skip-generate --skip-jepa --skip-actor --include-wireless
```

`--include-wireless` runs InF-SH scheduling only if the §15 validation gate reports `wireless_allowed: true`.

Root wrappers (`python scripts/train_jepa.py`, …) forward to the same `scripts/pipeline/` entry points.

## Step by step (fresh clone)

### 1. Validate the environment (plan §3)

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml
```

Write a JSON report:

```powershell
python scripts/pipeline/validate_environment.py --config configs/ts_jepa_dp_fixed.yaml --write-report
```

### 2. Generate trajectories (plan §4)

```powershell
python scripts/pipeline/generate_trajectories.py --config configs/ts_jepa_dp_fixed.yaml
```

Produces JEPA **200 / 40** and actor **100 / 20** train/test trajectories under `data_dp_fixed/trajectories/` (DP teacher, not uniform random). Actor trajectories use disjoint indices from JEPA (IC).

If sanity reports the wrong native size (not 128×256) or D_s/D_a overlap, delete `data_dp_fixed/trajectories/` and regenerate.

### 3. Train TS-JEPA (plan §10–§11)

Five seeds, cosine loss, Table II hparams, early stopping enabled (patience 20 / 20 val trajectories are IC):

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
```

Outputs `runs/ts_jepa_dp_fixed/seed_{0..4}/`, selected `runs/ts_jepa_dp_fixed/best.pt`, and `repetition_summary.json`.

Single seed / resume:

```powershell
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume-from runs/ts_jepa_dp_fixed/seed_2/last.pt
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --resume
```

### 4. Train the semantic actor (plan §12)

Do **not** wait for JEPA seeds 1–4 to debug the actor. Use seed-0’s encoder, then retrain against the selected 5-seed JEPA `best.pt`.

Debug on JEPA seed 0:

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 0 --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt
```

Final 5-seed actor training against the selected JEPA encoder:

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt
```

One actor seed at a time:

```powershell
python scripts/pipeline/train_actor.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 0 --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt
```

Outputs `runs/semantic_actor_dp_fixed/seed_{0..4}/` and `runs/semantic_actor_dp_fixed/best.pt` (best validation MSE).

### 5. Closed-loop evaluation (plan §14–§15)

`--jepa-checkpoint` is used when provided; otherwise the JEPA path stored in actor metadata is used.

Full §15 report (t-SNE, consecutive-frame MAPE Eq. 26, Fig. 4 sampling-rate MAPE, actor NMAE, 15-step prediction NMAE Eq. 27, control score Eq. 28, communication bits, packet-loss inference check):

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

Subset modes:

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode nmae --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode tsne --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode closed_loop_diagnostic --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --mode fig4 --out-dir runs/eval
```

Debug actor while JEPA seeds 1–4 are still running:

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --jepa-checkpoint runs/ts_jepa_dp_fixed/seed_0/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/seed_0/best.pt --out-dir runs/eval/sam_debug_seed0
```

Plan §15 checks in the baseline report:

| Check | Paper metric |
|-------|----------------|
| Embedding quality | t-SNE of JEPA test embeddings (Fig. 5) |
| Consecutive-frame MAPE | Eq. (26) at \(\tau_o=1\) ms; zero-denominator pixels skipped (IC) |
| Fig. 4 sampling-rate MAPE | Eq. (26) at IC rates `{1,2,5,10}` ms, with and without augmentation |
| Actor NMAE | encoded \(z\) → \(C_\varepsilon\) vs teacher \(u\), denorm, /40; mean-command baseline is a diagnostic IC |
| 15-step prediction | Eq. (27) NMAE for \(h=1\ldots K_p\) (\(K_p=15\)) |
| Closed-loop control | Eq. (28): \(R=1\) iff \(\lvert x-x_d\rvert\le 0.05\) and \(\lvert\vartheta\rvert\le 0.05\); accuracy = fraction of steps; **5 repeats, report best** |
| Communication | \(64\times128\times3\times8\) vs \(256\times8\) bits \(\approx 98.95\%\) reduction (not \(256\times32\)) |
| Stability | §14 inference: full receive vs intermittent loss (loss mask is IC) |

The band **[0.74, 1.0]** is for scalability plots only. It is **not** a pass/fail gate.

Do not pass `--include-wireless` until the report shows `wireless_allowed: true`.

### 6. Wireless / scheduling comparisons (plan §13, after §15)

`eval_runtime.py --mode wireless` runs **TS-JEPA** (predict on miss) under channel-aware / round-robin / opportunistic at SNR **5 / 10 / 20 dB**. That is the proposed controller under those schedulers (Fig. 10(a)), not the §16 conventional-control baselines.

Paper §16 round-robin / opportunistic **with conventional control and hold-last-command** are in the next section.

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode baseline --include-wireless --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode wireless --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

```powershell
python scripts/pipeline/eval_runtime.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --mode wireless --force-wireless --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --out-dir runs/eval
```

Closed-loop inference (plan §14): encoder \(\Psi_\theta\) on the device; predictor \(P_\varphi\) and actor \(C_\varepsilon\) remote. Received packet: \(x \to z \to \tilde u\) (z-score inverted to Newtons). Lost packet: roll \(P_\varphi\) with the last predicted command, then actor. Weights are not updated. First loss with no latent yet → `0.0` N (IC). Plant clip to \([-20,20]\) N is IC.

### 7. Paper baselines and reported experiments (plan §16–§17)

Control baselines (Section IV.C): nonlinear DP on the received observation; supervised RGB\(\to\)command at \(\kappa=2\) and \(\kappa=4\); generative autoencoder then nonlinear DP. Scheduling baselines use **conventional** control and **hold the last command** when unscheduled — not TS-JEPA prediction.

Supervised / AE layer sizes, AdamW, epochs, Fig. 7 embedding-axis values, Fig. 8 trajectory counts, Fig. 10 device grid, and Fig. 11 extra packet-loss rates are **implementation choices** (`docs/IMPLEMENTATION_CHOICES.md`; plan §18). Plan-specified: supervised \(\kappa\in\{2,4\}\), SNR \(\{5,10,20\}\) dB, hold-last when unscheduled, Fig. 6 no-prediction vs 15-step single TX.

**Train supervised \(\kappa=2\) and \(\kappa=4\)**

```powershell
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 4
```

**Train generative autoencoder (\(\kappa=2\))**

```powershell
python scripts/pipeline/train_autoencoder.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2
```

**Nonlinear DP closed loop (baseline 1)**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment dp --out-dir runs/eval/paper_experiments
```

**Fig. 6 — no-prediction vs 15-step single initial transmission**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig6 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt --supervised-kappa4 runs/baselines/supervised_kappa4/seed_0/best.pt --autoencoder-checkpoint runs/baselines/autoencoder_kappa2/seed_0/best.pt
```

**Fig. 7 — embedding-dimension grid (trains JEPA + actor per dim)**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig7 --out-dir runs/eval/paper_experiments
```

**Fig. 8 — training-set size (IC grid 50/100/150/200 of \(D_s\))**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig8 --out-dir runs/eval/paper_experiments
```

**Fig. 9 — target SNR 5 / 10 / 20 dB (TS-JEPA, single device)**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig9 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt
```

**Fig. 10 — scalability vs SNR (TS-JEPA predict vs supervised hold-last)**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
```

**Fig. 11 — scalability vs packet loss (extra drop rate is IC)**

```powershell
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment fig11 --snr 10 --out-dir runs/eval/paper_experiments --jepa-checkpoint runs/ts_jepa_dp_fixed/best.pt --actor-checkpoint runs/semantic_actor_dp_fixed/best.pt --supervised-kappa2 runs/baselines/supervised_kappa2/seed_0/best.pt
```

Smoke (short epochs) while debugging:

```powershell
python scripts/pipeline/train_supervised.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --kappa 2 --epochs 1 --max-train-trajectories 4
python scripts/pipeline/run_paper_experiments.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --experiment train_fig7 --jepa-epochs 1 --actor-epochs 1 --out-dir runs/eval/paper_experiments
```

## Tests

```powershell
pytest
```

## Repository layout

```
TS-JEPA/
├── configs/                 # YAML (baseline + experiment overlays)
├── data_dp_fixed/           # Generated trajectories (after step 2)
├── docs/                    # Plan, implementation choices
├── runs/                    # Checkpoints and evaluation outputs
├── scripts/pipeline/        # Main CLI
├── src/ts_jepa/             # Installable package
└── tests/
```

| File | Purpose |
|------|---------|
| `configs/ts_jepa_baseline.yaml` | Paper-faithful hyperparameters and plan asserts |
| `configs/ts_jepa_dp_fixed.yaml` | Overlay: `data_dp_fixed/` paths and run directory names |

See [`scripts/README.md`](scripts/README.md) for diagnose/dev helpers.

## Key references

- [`docs/plan.md`](docs/plan.md) — section-by-section implementation plan
- [`docs/IMPLEMENTATION_CHOICES.md`](docs/IMPLEMENTATION_CHOICES.md) — executable choices the paper does not specify
