# Implementation Choices

This file documents every detail required to run the baseline that is **not**
explicitly paper-specified in `docs/plan.md`.

Do not treat these as paper facts.

## Environment (plan §3)

Aligned with `docs/plan.md` §3; verified by `assert_plan_environment_config()` and
`validate_plan_environment()` (gate before dataset generation / training).

| Item | Value |
|------|-------|
| Backend | Custom `InvertedCartPoleEnv` (Gym vs custom is NOT SPECIFIED) |
| Observation | RGB frames (paper-specified) |
| Native render | `128 × 256 × 3` (IC; paper only specifies resize **to** 64×128) |
| Encoder input | `64 × 128 × 3` after 5×5 Gaussian-kernel resample (Algorithm 1) |
| κ context packing | supervised/AE only: `channel_concat` → `[3κ, 64, 128]` |
| Control | scalar `u ∈ [-20, +20]` N (paper-specified) |
| Sampling period τ_o | `1` ms (`dt = 0.001` s), 100 stored steps (paper-specified) |
| Dataset observation stride | `1` physics step per stored sample (paper setup; Fig. 4 does not change τ_o) |
| Appearance | per-trajectory cart/pole palette (IC; Fig. 5(c) is illustration, not a required sampler) |
| Environment validation | force limits, integration, pendulum dynamics, cart tracking, RGB render, stability, state/render consistency |

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
| Raw render size | `128 × 256` | IC camera; paper resize destination is 64×128 |
| Renderer | coverage anti-aliased float rasterizer | paper-silent; integer PIL made 1 ms motion invisible |
| Dataset stride | 1 physics tick / stored frame | paper τ_o = 1 ms, 100 steps; Fig. 4 is analysis only |

## Datasets D_s / D_a (plan §4)

Paper specifies counts (200/40 and 100/20) and that D_a embeddings are formed **after** TS-JEPA using Ψθ. Whether the underlying RGB rollouts are shared is **NOT SPECIFIED**.

| Choice | Value | Notes |
|--------|-------|-------|
| Physical rollouts | disjoint `trajectory_index` / seed | D_s: 0–239; D_a: 240–359. Actor test is not a subset of JEPA train |
| Native stored RGB | `128 × 256 × 3` | IC camera; encoder input is 64×128 after Gaussian resize |
| Stale `data_dp_fixed/` | regenerate | older files stored at 64×128 with overlapping D_s/D_a indices will fail sanity |

## Fig. 4 sampling-rate MAPE (plan §15)

Fig. 4 compares consecutive-frame MAPE (Eq. 26) at different sampling rates, with and without augmentation. The rate axis is **NOT SPECIFIED**. Main simulation τ_o stays 1 ms.

| Choice | Value |
|--------|-------|
| Rate grid | `{1, 2, 5, 10}` ms (`experiments.fig4_sampling_interval_ms`) |
| How | integer subsample of stored 1 ms frames |
| With augmentation | color jitter + color drop on RGB (no ImageNet normalize / resize) |
| CLI | `python scripts/pipeline/eval_runtime.py --mode fig4` (no checkpoints) |

## Actor mean-command baseline (diagnostic)

Not a paper metric. Actor training and `evaluate_actor_nmae` record MSE/NMAE of predicting the training-set mean command so a collapsed actor is visible. This does **not** fail `assert_plan_*`.

## LoS probability (Eq. 5)

Paper typesets a product of fractions that drives P_LoS → 0 at Table IV geometry. This repo uses the cited InF-SH / 3GPP form with the height ratio **inside** k (see `WirelessChannelModel.los_probability`). That reading is an implementation reconciliation, not a recovered paper number.


## DP teacher

| Choice | Value | Notes |
|--------|-------|-------|
| Method | discretized finite-horizon value iteration + local action refinement | paper requires nonlinear DP and Eq. 3 is undiscounted; grids/`R`/`K` unspecified |
| Dataset control source | `dp_nonlinear` (recorded in each `.npz`) | plan §4.1 — **not** `Uniform(-20, 20)` |
| Post-generation verification | `verify_not_uniform_random_actions()` | peaked histogram **or** lag-1 autocorr vs uniform-iid (1 ms DP may lack persistence) |
| Physics `dt` | `0.001` s | matches simulation sampling |
| DP substeps `N` | `1` | PAPER-IMPLIED: one Bellman transition = one \(\tau_o\) (Eq. 2 subject to Eq. 1). Not an IC. |
| Value interpolation | nodal Taylor \(V(s_g)+∇V·(s'−s_g)\) | IC. At 1 ms, ±20 N moves \(\dot\theta\) by ~0.06 rad/s vs 0.6 rad/s grid spacing, so nearest-neighbor maps every force to one cell and \(u=0\) wins on \(\tfrac12 R u^2\). Multilinear interpolation still fails: convex \(V\) plus a velocity node at 0 creates a kink, so any 1 ms motion looks costly. Central-difference \(∇V\) recovers the mixed partial that prefers restoring \(u\). Grid is **not** refined to manufacture NN cell splits (that would need ~0.003 rad/s \(\dot\theta\) bins). |
| Stage cost | `½‖s_{k+1}−s★‖² + ½ R u²` | one-step state cost; control charged **once** per DP decision |
| Control effort weight `R` | `0.001` | scalar stand-in for positive-definite `R` |
| Init state noise | `±0.35` (±0.175 on θ) | IC — wide enough to leave π≈0 deadzone under Ns=0 |
| Discount | `1.0` (finite \(K\)) | Eq. 3 has no \(\gamma\). Do not keep \(0.99\) per 1 ms step: that horizon is ~100 ms and restoring \(Q\) loses to \(\tfrac12 R u^2\). |
| Value-iteration iters \(K\) | `300` | IC finite horizon \(\approx 300\) ms \(\approx 1.3/\omega\). 50 backups was leftover from the 50 ms Bellman and only looks 50 ms ahead at 1 ms. |
| Force bins | `11` in `[-20,20]` | unspecified discretization |
| State grids | `x×13`, `ẋ×11`, `θ×13`, `θ̇×11` (see YAML) | unspecified; **not** refined to “see” 1 ms NN cells (that would need ~0.01 rad/s \(\dot\theta\) bins) |
| `act()` | same one-step Bellman cost as VI | local discrete re-opt over table ± neighbors |

## Input tensor construction

Aligned with `docs/plan.md` §6; verified by `assert_plan_temporal_config()`.

| Choice | Value |
|--------|-------|
| κ | `2` consecutive frames for **supervised/AE**; TS-JEPA uses current frame only |
| Kp | `15` prediction horizon |
| Embedding dim | `256` |
| Context at step k | `x_k` (one RGB frame, Algorithm 1) |
| Control sequence | `[u_k, ..., u_{k+Kp-1}]` (teacher DP commands) |
| Target inputs | `x_{k+1} .. x_{k+Kp}` (one RGB frame each) |
| Predictor outputs | `[z̃_{k+1}, ..., z̃_{k+Kp}]` autoregressively |

## JEPA loss (plan §10)

Aligned with `docs/plan.md` §10; verified by `assert_plan_jepa_loss_config()`.

| Item | Value |
|------|-------|
| Objective | cosine similarity between predicted and target embeddings |
| Per-step | `cos_sim = (z̃·z̄) / (‖z̃‖₂ ‖z̄‖₂)` |
| Paper loss | `L_JEPA = -(1/Kp) Σ_j cos_sim(z̃_j, z̄_j)` |
| Training implementation | `jepa_loss()` = `-mean(cos_sim)` over batch and horizon |
| Equivalent minimizing form | `cosine_alignment_loss()` = `1 - mean(cos_sim)` (same gradients) |
| Targets | EMA target encoder, stop-gradient (`encode_targets`) |

## JEPA training hyperparameters (plan §11)

Aligned with `docs/plan.md` §11; verified by `assert_plan_jepa_training_config()`
in baseline entry scripts (not inside the train loop — smoke tests may shrink batch/Kp).

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD | Table II; momentum `0.0` is IC (not a paper number) |
| Learning rate | `0.2` |
| LR warmup | **10 epochs**, linear `0.02 → 0.2` (IC; Table II silent). Peak LR stays 0.2. |
| Batch size | `256` (effective; microbatch accumulation is IC) |
| Epochs | `150` |
| Weight decay | `0.0004` on non-BN weights; **BN affine WD = 0** (IC) |
| Grad clip | `1.0` (IC; Table II silent). If EMA still desyncs after warmup, try `0.5` or `0.1`. |
| EMA decay η | `0.99` |
| LR schedule | warmup, then ×`0.99` every `20` epochs (decay is paper; warmup is IC) |
| Kp / κ / emb | `15` / `2` / `256` |
| Forbidden | Adam, LR=`0.001`, EMA=`0.996`, wd=`1e-5`, epochs=`200` |

## JEPA training procedure (plan §10 Algorithm 1)

Aligned with `docs/plan.md` §10; implemented by `jepa_forward_batch()` +
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

Paper-specified (plan §7–§8): widths 64→128→256 with BN+ReLU; target = copy, stop-grad, EMA η=0.99.

Everything below is IC (plan §7: do not label stem / MaxPool / GAP / 4×8 as paper architecture).

| Component | Value |
|-----------|-------|
| Context encoder Ψθ | `ContextEncoder` |
| Target encoder Ψθ̄ | same class (`TargetEncoder = ContextEncoder`), init `θ̄ ← θ` |
| Channel widths | 64 → 128 → 256 (paper) |
| Embedding dim | 256 (paper-implied from predictor output; Fig. 7 also lists 64/128/350) |
| Stem | `7×7` conv stride-2 → BN → ReLU → MaxPool3×3 stride-2 (IC) |
| Stage 64 | residual, stride 1 (IC) |
| Stage 128 | residual, stride 2 (IC) |
| Stage 256 | residual, stride 2 (IC) |
| Head | spatial pool `4×8` → flatten → linear → **L2-normalize** (IC; not GAP) |
| Blocks per stage | `2` (IC) |
| Embedding L2-norm | `F.normalize` on Ψ outputs (IC). Paper Eq. (13) is cosine / direction-only; unbounded z + SGD 0.2 makes 15-step AR oscillate. |
| Target trainable | `false` — frozen, stop-gradient forward (paper) |
| Target update | EMA `θ̄ ← η θ̄ + (1-η) θ`, η=0.99 (paper) |
| BN running-stat buffers | same EMA as θ (float buffers); integer buffers copied |

## Predictor command source (plan §9, OPEN)

Plan §9: paper uses ũ notation during predictor training, but the
pretraining command-generation mechanism is **not specified**. Status remains
`OPEN`. The baseline keeps `teacher_dp` as a documented candidate and does
**not** invent a paper-exact mechanism.

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

Paper-specified: MLP hidden **1024**, output **256**, autoregressive, command-conditioned. No virtual-channel inputs.

Hidden ReLU, concat, input width 257, and hidden BatchNorm1d are IC.

| Component | Value |
|-----------|-------|
| Type | MLP |
| Stack | `Linear(257→1024) → BatchNorm1d(1024) → ReLU → Linear(1024→256) → L2-normalize` (ReLU, concat, BN, and L2 are IC) |
| Hidden BN | `BatchNorm1d` after the 1024-wide linear (IC). Conditions predictor grads under Table II SGD 0.2; not a paper layer. WD on this BN affine is 0 (same IC as encoder BN). |
| Autoregressive | `z_{j+1} = normalize(Pφ(concat(z_j, u_j)))`; next input is `z_{j+1}.detach()` |
| AR truncation | Detach between steps (IC). Paper Eq. (12) is still unrolled. Eq. (13) writes one-step cosine `ẑ_{k+1}`; 15-step BPTT is not specified and is unstable with Table II SGD 0.2. Encoder cosine matches the first step; Pφ is trained at every horizon step. |
| Inputs | embedding + scalar control only |
| Forbidden | virtual inputs / channel variables / channel embeddings |

## Preprocessing pipeline order

Aligned with `docs/plan.md` §5; verified by `assert_plan_preprocessing_config()`.

| Split | Order |
|-------|-------|
| Train | jitter (random op order) → color drop → ImageNet normalize → 5×5 Gaussian-kernel resample to 64×128 (σ ~ U[0.1, 0.2]) |
| Eval | ImageNet normalize → 5×5 Gaussian-kernel resample to 64×128 (σ = 0.15 midpoint); no stochastic aug |
| Commands | z-score on JEPA train commands for **Pφ** only: `u_norm = (u - μ) / σ` |

## Semantic actor output

| Choice | Value |
|--------|-------|
| Network output | linear, **physical Newtons** (Eq. 15) |
| Runtime | clip to `[-20, +20]` N; z-score that force only when feeding Pφ |

## Inference / evaluation (plan §14–§15)

| Choice | Value | Notes |
|--------|-------|-------|
| First lost packet before any embedding | command `0.0` N | paper does not specify the empty-latent case |
| Plant force clip | `[u_min, u_max]` after denorm | env also clips; not an actor nonlinearity |
| MAPE zero denominator (Eq. 26) | skip pixels with \(x_{k-1}(υ)=0\) | paper silent |
| Embedding bit-width | 8 bits/value | recovered from the paper’s **98.95%** reduction (`64×128×3×8` vs `256×8`); not an invented `256×32` |
| t-SNE perplexity / sample cap | sklearn default-style perplexity; `tsne_max_samples=500` | visualization only |
| Synthetic packet-loss mask (stability) | warm-up then alternate receive/loss | paper does not specify this diagnostic pattern |
| `[0.74, 1.0]` control band | recorded, not a pass/fail gate | plan: scalability plots only |
| `--jepa-checkpoint` | honored when passed; else actor-metadata path | CLI convenience |

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
| Eval artifacts | `runs/eval/{baseline,nmae,closed_loop,embedding_tsne,baseline_validation}_*` + PNG plots | wireless only after baseline validation `PASS` |
| Prediction NMAE | `mean(\|u_pred-u_true\|)/40` after denorm | plan §15 Eq. (27); not `mean(\|err\|)/mean(\|u\|)` |
| Wireless delivery | `packet_received = scheduled AND γ≥γ_th` | controller path; AoI follows α (plan §13 Eq. 17), not delivery |
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

Lyapunov \(B\) in eq. (22) is **omitted** (paper: does not affect performance).
It is not an IC number to recover.

### Still IMPLEMENTATION CHOICE (no numerical value in the paper)

| Choice | Value |
|--------|-------|
| Max scheduled devices `J` | `2` |
| Device count `I` | `4` |
| `p_max` | `0.2` W |
| Drift-plus-penalty `V` | `1.0` |
| AoI threshold `β_th` | `5.0` |

Hall size / room height / bandwidth are stored from Table IV; path-loss uses the
paper distance/height/clutter formulas above (hall polygon layout not simulated).

## Paper baselines / Figs. 6–11 (plan §16–§17)

| Item | Paper | Implementation choice |
|------|-------|------------------------|
| DP on received observation | nonlinear DP | DP uses privileged **4D plant state** (cannot run on RGB) |
| Supervised \(\kappa\in\{2,4\}\) | RGB → command | ResNet 64/128/256 + MLP 1024→256→1; AdamW lr \(10^{-3}\), batch 64, 50 epochs |
| Generative AE \(\kappa=2\) | reconstruct state, then nonlinear control | same ResNet encoder; linear RGB decoder; **4D state head** + DP; recon weights 1.0/1.0 |
| RR / opportunistic miss | hold last command | initial force `0.0` N before any delivery |
| Fig. 6 miss fallbacks | 15-step case: conventional hold (plan: last command if unscheduled) | zero-action extra; not \(P_\varphi\) |
| Fig. 4 sampling rates | experiment exists | `{1,2,5,10}` ms subsample of stored τ_o=1 ms frames |
| Fig. 7 embedding dims | experiment exists | `{64,128,256,350}` in YAML — **not** in plan.md |
| Fig. 8 train sizes | experiment exists | `{50,100,150,200}` of \(D_s\) |
| Fig. 9 SNR | `{5,10,20}` dB | **eval-time** SNR; training-time dropout not specified |
| Fig. 10 device counts \(I\) | scalability vs RR/opp | `{2,4,8,16,32}`; \(J\) stays IC |
| Fig. 11 extra packet loss | “including packet loss” | Bernoulli `{0,0.1,0.2,0.3}` after a successful scheduled slot; hold on outage is IC |
| Acceptable score band | `[0.74, 1.0]` (plan §15 plots) | used only to count supported devices in Figs. 10–11 |

## Out of scope

- GE-JEPA
- Gilbert-Elliott masking
- Burst Position Encoding
- Burst-aware / packet-loss-conditioned neural architecture changes
