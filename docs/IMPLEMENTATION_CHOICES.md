# Implementation Choices

This file documents every detail required to run the baseline that is **not**
explicitly paper-specified in `docs/TS-JEPA_Paper-Faithful_Final.md`.

Do not treat these as paper facts.

## Environment (plan §3)

Aligned with `docs/plan.md` §3; verified by `assert_plan_environment_config()` and
`validate_plan_environment()` (plan §3.3 gate before dataset generation / training).

| Item | Value |
|------|-------|
| Backend | Custom `InvertedCartPoleEnv` — **not** Gym CartPole-v1 |
| Observation | RGB frames (not raw 4D state to the encoder) |
| Render resolution | `64 × 128 × 3` |
| κ context | `2` frames → `[6, 64, 128]` via `channel_concat` |
| Control | scalar `u ∈ [-20, +20]` N |
| Sampling period τ_o | `1` ms (`dt = 0.001` s) |
| §3.3 validation | force limits, integration, pendulum dynamics, cart tracking, RGB render, stability, state/render consistency |

## Environment / physics

| Choice | Value | Notes |
|--------|-------|-------|
| Cart mass `M` | `1.0` kg | unspecified by paper |
| Pole mass `m` | `0.1` kg | unspecified by paper |
| Pole half-length `l` | `0.5` m | unspecified by paper |
| Gravity `g` | `9.81` m/s² | unspecified by paper |
| Track limit | `2.4` m | renderer/world bound |
| Integrator | semi-implicit Euler | unspecified by paper |
| Process noise `N_s` | `0` (std=`0.0`) | paper leaves cart-pole `N_s` unspecified; deterministic baseline |
| Desired state | `[0,0,0,0]` | upright at origin |
| Raw render size | `64 × 128` | plan §3.1 target rendering resolution |

## DP teacher

| Choice | Value | Notes |
|--------|-------|-------|
| Method | discretized discounted value iteration + local action refinement | paper requires nonlinear DP; grids/`R` unspecified |
| Dataset control source | `dp_nonlinear` (recorded in each `.npz`) | plan §4.1 — **not** `Uniform(-20, 20)` |
| Post-generation verification | `verify_not_uniform_random_actions()` | lag-1 autocorr + histogram structure vs uniform-iid reference |
| Physics `dt` | `0.001` s | matches simulation sampling |
| DP substeps `N` | `50` | one Bellman transition = **0.05 s** held force |
| Stage cost | `Σ_{j=1..N} ½‖s_j−s★‖² + ½ R u²` | integrated state cost; control charged **once** per DP decision |
| Control effort weight `R` | `0.001` | scalar stand-in for positive-definite `R` |
| Init state noise | `±0.35` (±0.175 on θ) | IC — wide enough to leave π≈0 deadzone under Ns=0 |
| Discount | `0.99` | unspecified |
| Value-iteration iters | `50` | unspecified |
| Force bins | `11` in `[-20,20]` | unspecified discretization |
| State grids | `x×13`, `ẋ×11`, `θ×13`, `θ̇×11` (see YAML) | unspecified |
| `act()` | same N-step Bellman cost as VI | local discrete re-opt over table ± neighbors |

## Input tensor construction

Aligned with `docs/plan.md` §8; verified by `assert_plan_temporal_config()`.

| Choice | Value |
|--------|-------|
| κ | `2` consecutive context frames |
| Kp | `15` prediction horizon |
| Embedding dim | `256` |
| Context at step k | κ frames ending at k → `[x_k-1, x_k]` |
| Control sequence | `[u_k, ..., u_{k+Kp-1}]` (teacher DP commands) |
| Target inputs | κ-windows ending at `k+1 .. k+Kp` |
| Predictor outputs | `[z̃_{k+1}, ..., z̃_{k+Kp}]` autoregressively |

## JEPA loss (plan §11)

Aligned with `docs/plan.md` §11; verified by `assert_plan_jepa_loss_config()`.

| Item | Value |
|------|-------|
| Objective | cosine similarity between predicted and target embeddings |
| Per-step | `cos_sim = (z̃·z̄) / (‖z̃‖₂ ‖z̄‖₂)` |
| Paper loss | `L_JEPA = -(1/Kp) Σ_j cos_sim(z̃_j, z̄_j)` |
| Training implementation | `jepa_loss()` = `-mean(cos_sim)` over batch and horizon |
| Equivalent minimizing form | `cosine_alignment_loss()` = `1 - mean(cos_sim)` (same gradients) |
| Targets | EMA target encoder, stop-gradient (`encode_targets`) |

## JEPA training hyperparameters (plan §12)

Aligned with `docs/plan.md` §12; verified by `assert_plan_jepa_training_config()`
in baseline entry scripts (not inside the train loop — smoke tests may shrink batch/Kp).

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum 0) |
| Learning rate | `0.2` |
| Batch size | `256` (effective; microbatch accumulation is IC) |
| Epochs | `150` |
| Weight decay | `0.0004` |
| EMA decay η | `0.99` |
| LR schedule | ×`0.99` every `20` epochs |
| Kp / κ / emb | `15` / `2` / `256` |
| Forbidden | Adam, LR=`0.001`, EMA=`0.996`, wd=`1e-5`, epochs=`200` |

## JEPA training procedure (plan §13)

Aligned with `docs/plan.md` §13; implemented by `jepa_forward_batch()` +
`jepa_sgd_and_ema_step()`; verified by `assert_plan_jepa_procedure_config()`.

| Step | Action |
|------|--------|
| 1 | Context encoding `z = Ψθ(x)` |
| 2 | Target encoding `z̄ = Ψθ̄(x_future)` with stop-gradient |
| 3 | Autoregressive prediction with §10 command source |
| 4 | `L_JEPA = -mean(cos_sim)` |
| 5 | SGD update of θ (context) and ϕ (predictor) only |
| 6 | EMA `θ̄ ← η θ̄ + (1-η) θ` after optimizer step |

## Encoder ResNet low-level structure

Aligned with `docs/plan.md` §6–§7; verified by `assert_plan_encoder_config()`.

| Component | Value |
|-----------|-------|
| Context encoder Ψθ | `ContextEncoder` |
| Target encoder Ψθ̄ | same class (`TargetEncoder = ContextEncoder`), init `θ̄ ← θ` |
| Channel widths | 64 → 128 → 256 |
| Embedding dim | 256 |
| Stem | `7×7` conv stride-2 → BN → ReLU → MaxPool3×3 stride-2 |
| Stage 64 | residual, stride 1 |
| Stage 128 | residual, stride 2 |
| Stage 256 | residual, stride 2 |
| Head | global average pool → linear |
| Blocks per stage | `2` (plan §26.2 VERIFY) |
| Target trainable | `false` — frozen, stop-gradient forward |
| Target update | EMA `θ̄ ← η θ̄ + (1-η) θ`, η=0.99 |
| BN running-stat buffers | copied from online each EMA step (IC) |

## Predictor §10 command-resolution (OPEN)

Plan §10 / §26.1: paper uses ũ notation during predictor training, but the
pretraining command-generation mechanism is **not fully specified**. Status remains
`OPEN` / `PENDING_PAPER_IMPLEMENTATION_RESOLUTION`.

| Item | Value |
|------|-------|
| Status | `OPEN` (must not be closed without paper/source evidence) |
| Selected candidate | `teacher_dp` |
| Paper-exact? | **No** — explicitly `paper_exact: false` in config |
| Training conditioning | `teacher_commands_norm` = DP trajectory ground-truth `u*` |
| Inference (packet lost) | `semantic_actor` last command (closed loop) |
| Forbidden | labeling `teacher_dp` as paper ũ; virtual-channel inputs |
| Other candidates | `semantic_actor`, `sequential_actor_predictor`, `recovered_from_source` — not implemented |

Recorded in: `configs/ts_jepa_baseline.yaml` → `predictor_command_resolution`,
JEPA checkpoints, `repetition_summary.json`, eval reports.

## Predictor architecture (plan §9)

Aligned with `docs/plan.md` §9; verified by `assert_plan_predictor_config()`.

| Component | Value |
|-----------|-------|
| Type | MLP |
| Stack | `Linear(257→1024) → ReLU → Linear(1024→256)` |
| Autoregressive | `z_{j+1} = Pφ(concat(z_j, u_j))` |
| Inputs | embedding + scalar control only |
| Forbidden | virtual inputs / channel variables / channel embeddings |

## Preprocessing pipeline order

Aligned with `docs/plan.md` §5; verified by `assert_plan_preprocessing_config()`.

| Split | Order |
|-------|-------|
| Train (plan §5.1) | augmentation (color jitter → color drop) → ImageNet normalize → formatting (Gaussian blur → resize to 64×128) |
| Eval (plan §5.2) | resize → ImageNet normalize (no stochastic aug, no blur) |
| Commands (plan §5.3) | z-score on JEPA train commands: `u_norm = (u - μ) / σ`; denorm `u = u_norm * σ + μ` |

## Semantic actor output

| Choice | Value |
|--------|-------|
| Network output | linear (normalized command domain) |
| Runtime | `u = u_norm * σ + μ`, then clip to `[-20, +20]` N |

## Training batching / GPU memory

| Choice | Value | Notes |
|--------|-------|-------|
| Effective JEPA batch | `256` | paper-specified; one `optimizer.step()` / EMA update per 256 samples |
| Microbatch size | `16` (default IC) | forward/backward chunk size for 8 GB GPUs; try `32` if VRAM allows |
| Accumulation | `batch_size / microbatch_size` | loss scaled by `1/accumulation` so grads match mean over 256 |
| Leftover micros | dropped | e.g. 956 micros → 59×16 used, 12 leftover never `optimizer.step()` |
| Target-encoder chunking | `256` frames | eval/no-grad; BN running stats; does not change outputs |
| BatchNorm | unchanged ResNet BN | BN stats still update on microbatches in train mode (no SyncBN / GhostBN) |

## Laptop runtime (Windows + RTX 5070 8 GB)

| Choice | Value | Notes |
|--------|-------|-------|
| DataLoader `num_workers` | `4` on CUDA, `0` on CPU | overlaps aug/preprocess with GPU; CPU/tests stay single-process |
| `pin_memory` + CUDA prefetch | on | hides H2D copy latency |
| `dataloader_timeout_s` | `120` | PyTorch worker queue timeout; raises instead of silent mid-epoch hang |
| `heartbeat_s` / `stall_timeout_s` | `30` / `180` | stdout + `runs/*/train.log` heartbeats; STALL lines when progress stops |
| `cudnn.benchmark` | on | fixed 64×128 shapes |
| Checkpoint I/O | CPU `state_dict` + `cuda.synchronize` before `torch.save` | mitigates WDDM access-violation crashes mid-epoch |
| Loss `.item()` | once per effective step | avoids forcing a GPU sync every microbatch |
| Eval artifacts | `runs/eval/{baseline,nmae,closed_loop,embedding_tsne,baseline_validation}_*` + PNG plots | wireless only after baseline validation passes |
| Checkpoint resolve | `best.pt` else `seed_0/best.pt` | single-seed prelim vs 5-seed selected `best.pt` |
| Experiment overlay | `configs/ts_jepa_dp_fixed.yaml` | `_base` merge; `data_dp_fixed` + `ts_jepa_dp_fixed` / `semantic_actor_dp_fixed` run dirs |

## Validation / early stopping / repetitions

| Choice | Value | Notes |
|--------|-------|-------|
| JEPA val trajectory count | `20` held out from the 200 **train** trajectories | paper enables ES but does not specify count |
| JEPA patience | `20` epochs | unspecified |
| Actor val | `20%` holdout from actor **train** embeddings | test trajectories untouched |
| Actor patience | `30` epochs | unspecified |
| JEPA/actor test sets | 40 / 20 trajectories | used only for post-hoc reporting, never model selection |
| Repetition seeds | `{0,1,2,3,4}` | paper requires 5 reps + best result; seed values unspecified |
| Best-run criterion (JEPA) | lowest validation cosine-alignment loss | paper-compatible validation performance |
| Best-run criterion (actor) | lowest validation MSE | paper-compatible validation performance |

## Wireless / scheduler (networking layer)

### Paper-specified (implemented)

- InF-SH path loss eqs. (4)–(7), LoS probability (5), Rayleigh `|H|^2`
- SNR model (8), required power (23)
- AoI update (17), virtual queue (18)
- Algorithm 2 feasibility / Top-J flow
- Table IV geometry/radio scalars and `γ_th ∈ {5,10,20}` dB
- Shadow fading std `4.0` dB on general PL model

### Index selection reconciliation (documented choice)

Paper eq. (24) **minimizes** `Σ α_i C_i` with

`C_i = 1 - (β+1)^2 - 2 Q β + V p_req` (same expression as typeset eq. 25).

Algorithm 2 selects the **largest positive** indices. The code therefore uses

`U_i = -C_i`

and selects Top-J devices with `U_i > 0`. This preserves Algorithm 2’s selection
rule while remaining consistent with the minimization in (24).

### Still IMPLEMENTATION CHOICE (no numerical value in Final.md)

| Choice | Value |
|--------|-------|
| Max scheduled devices `J` | `2` |
| Device count `I` | `4` |
| `p_max` | `0.2` W |
| Drift-plus-penalty `V` | `1.0` |
| AoI threshold `β_th` | `5.0` |

Hall size / room height / bandwidth are stored from Table IV; path-loss uses the
paper distance/height/clutter formulas above (hall polygon layout not simulated).

## Out of scope until baseline validated

- GE-JEPA
- Gilbert-Elliott masking
- Burst Position Encoding
- Burst-aware / packet-loss-conditioned neural architecture changes
