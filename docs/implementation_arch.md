# TS-JEPA Implementation Architecture

This document describes the **current codebase implementation** of the paper-faithful TS-JEPA baseline: every module, parameter, tensor shape, method, and training rule as it exists in this repository.

It is an implementation dump, not a paper-faithfulness argument. Paper vs implementation-choice (IC) labels are included only so a report can distinguish them. Executable IC values live in `docs/IMPLEMENTATION_CHOICES.md`. Hyperparameters are loaded from `configs/ts_jepa_baseline.yaml` (typically via overlay `configs/ts_jepa_dp_fixed.yaml`).

**Package root:** `src/ts_jepa/`

---

## End-to-end dataflow

```text
InvertedCartPoleEnv (RGB 128×256, dt=1 ms)
        │  DPControlTeacher.act(state) → u* ∈ [-20, 20] N
        ▼
.npz trajectories  D_s (JEPA 200/40)  and  D_a (actor 100/20)
        │
        ├─ PreprocessPipeline  →  [3, 64, 128]
        │         │
        │         ▼
        │   ContextEncoder Ψθ  ──encode x_k──►  z_k ∈ R^256 (L2 unit)
        │         │
        │         │  EMA η=0.99, stop-grad
        │         ▼
        │   TargetEncoder Ψθ̄ ──encode x_{k+1:k+Kp}──►  z̄ ∈ R^{Kp×256}
        │         │
        │   Predictor Pφ(z, u_norm)  ──AR Kp=15──►  z̃ ∈ R^{Kp×256}
        │         │
        │         └─ L_JEPA = -mean cosine(z̃, z̄)  →  SGD on θ,φ
        │
        └─ frozen Ψθ on D_a frames  →  (z, u) pairs
                  │
                  ▼
            SemanticActor Cε(z) → ũ in Newtons
                  │  MSE vs teacher u
                  ▼
            FrozenRuntimeController (closed loop)
```

---

# 1. Environment

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/env/cartpole_ode.py` | Nonlinear ODE + integrator |
| `src/ts_jepa/env/cartpole_rgb.py` | Gym-like env: reset/step/render/rollout |
| `src/ts_jepa/env/renderer.py` | Coverage anti-aliased RGB rasterizer |
| `src/ts_jepa/env/factory.py` | `build_inverted_cartpole_env(config)` |
| `src/ts_jepa/env/env_validation.py` | Plan §3.3 validation suite |
| `src/ts_jepa/plan/environment.py` | `PLAN_ENVIRONMENT` + config asserts |
| `scripts/pipeline/validate_environment.py` | CLI gate before data generation |

**Config blocks:** `simulation.*`, `environment.*`

The backend is a **custom** `InvertedCartPoleEnv`, not OpenAI Gym `CartPole-v1`. Observations are RGB frames, not the 4-D plant state. The 4-D state is used only by the DP teacher and for scoring.

## 1.1 Plant state and control

State vector (always `float64`, shape `(4,)` or batched `(N, 4)`):

```text
s = [x, ẋ, θ, θ̇]
```

| Symbol | Meaning | Units |
|--------|---------|-------|
| `x` | Cart position (origin at track center) | m |
| `ẋ` | Cart velocity | m/s |
| `θ` | Pole angle; **θ = 0 is upright**; positive θ is counterclockwise | rad |
| `θ̇` | Angular velocity | rad/s |

Control `u` is a **scalar horizontal force** on the cart, clipped to `[-20, +20]` N in both `CartPoleODE.derivatives` / `step` and `InvertedCartPoleEnv.clip_force`.

Desired state (IC): `[0, 0, 0, 0]` — upright at the origin.

Control score (paper Eq. 28, used at eval, not as a training loss):

```text
R = 1  iff  |x − x_d| ≤ 0.05  AND  |θ| ≤ 0.05
R = 0  otherwise
```

Implemented as `InvertedCartPoleEnv.control_score(state, position_tol=0.05, angle_tol=0.05)`.

## 1.2 Physics parameters (all IC)

`CartPoleParams` (`cartpole_ode.py`), mirrored in `simulation.physics`:

| Parameter | Symbol in code | Value |
|-----------|----------------|-------|
| Cart mass | `cart_mass` `M` | `1.0` kg |
| Pole mass | `pole_mass` `m` | `0.1` kg |
| Pole half-length (COM) | `pole_length` `l` | `0.5` m |
| Gravity | `gravity` `g` | `9.81` m/s² |
| Track half-width (render/world) | `track_limit` | `2.4` m |

Sampling (paper-specified):

| Parameter | Value |
|-----------|-------|
| `τ_o` | `1` ms |
| `dt` | `0.001` s (`must equal sampling_interval_ms/1000`) |
| Stored steps per trajectory | `100` → duration `100 ms` |
| `observation_stride_steps` | `1` (one stored sample per physics tick) |
| Process noise `N_s` | `0.0` (deterministic baseline; paper `N_s` unspecified) |

## 1.3 Continuous-time dynamics

`CartPoleODE.derivatives(state, force)` implements the standard underactuated inverted-cart-pole equations (Barto/Sutton form). `θ = 0` is **upright**. Force is clipped to `±20` N before the equations.

Let `M_t = M + m`, `sθ = sin(θ)`, `cθ = cos(θ)`:

```text
temp      = (F + m · l · θ̇² · sθ) / M_t
θ̈_den     = l · (4/3 − m · cθ² / M_t)
θ̈         = (g · sθ − cθ · temp) / θ̈_den
ẍ         = temp − (m · l · θ̈ · cθ) / M_t
ṡ         = [ẋ, ẍ, θ̇, θ̈]
```

Supports a single state `(4,)` or a batch `(N, 4)`. The same vectorized path is used by the DP teacher’s Bellman backups.

## 1.4 Integrator

`CartPoleODE.step(state, force, process_noise_std=0.0)` — **semi-implicit Euler** (IC):

```text
ẋ_{k+1}  = ẋ_k  + dt · ẍ(s_k, F)
θ̇_{k+1}  = θ̇_k  + dt · θ̈(s_k, F)
x_{k+1}  = x_k  + dt · ẋ_{k+1}
θ_{k+1}  = θ_k  + dt · θ̇_{k+1}
```

If `process_noise_std > 0`, i.i.d. Gaussian noise is added after the Euler step. Baseline sets this to `0`.

## 1.5 `InvertedCartPoleEnv` API

Constructor defaults (overridden from YAML by `build_inverted_cartpole_env`):

| Arg | Config | Default / baseline |
|-----|--------|--------------------|
| `render_height` | `simulation.render_height` | `128` |
| `render_width` | `simulation.render_width` | `256` |
| `dt` | `simulation.dt` | `0.001` |
| `force_min` / `force_max` | `simulation.control_min_N` / `control_max_N` | `±20` |
| `desired_state` | `simulation.desired_state` | `[0,0,0,0]` |
| `process_noise_std` | `simulation.process_noise_std` | `0.0` |
| `init_noise` | `simulation.init_noise` | **`0.35`** (YAML; class default `0.05`) |
| `observation_stride_steps` | `simulation.observation_stride_steps` | `1` |

**`reset(seed, init_noise=None)`**

- RNG: `np.random.default_rng(seed)`
- Noise: `U[-noise_scale, +noise_scale]` on all four states
- Angle scaled by `0.5`: `noise[2] *= 0.5` so `θ ∈ [−0.175, +0.175]` when `init_noise=0.35`
- `state ← desired_state + noise`

**`step(force)`**

- Clip force → `ode.step` → `renderer.render(state)`
- Returns `(state_copy, frame)` with `frame` `uint8` `[H, W, 3]`

**`rollout(teacher, steps, seed, observation_stride_steps=None)`**

Per stored step `k` (paper setup: stride = 1):

1. `u*_k = clip(teacher.act(state_k))`
2. Store `frame_k = render(state_k)`, `u*_k`, `state_k`
3. Apply `u*_k` for `observation_stride_steps` physics ticks

Returns:

```text
frames:   uint8   [T, H, W, 3]     T=100, H=128, W=256
commands: float32 [T]
states:   float64 [T, 4]
```

## 1.6 RGB renderer (`CartPoleRenderer`)

Native camera size is **IC**: `128 × 256 × 3` uint8. The paper only specifies the encoder destination `64 × 128`. Integer-pixel PIL rendering made 1 ms motion invisible; this renderer uses **float world-to-pixel coordinates + coverage anti-aliasing**.

**World-to-pixel**

```text
world_half = track_limit = 2.4 m
px_per_m   = (width · 0.8) / (2 · world_half) = 256·0.8 / 4.8 ≈ 42.667 px/m
origin_x   = width / 2
rail_y     = height · 0.72
cart_x     = origin_x + x · px_per_m
cart_w     = max(12, 0.35 · px_per_m) ≈ 14.93
cart_h     = max(8,  0.2  · px_per_m) ≈ 8.53
pole_len   = pole_length · 2 · px_per_m = 1.0 · px_per_m ≈ 42.67 px
tip        = (cart_x + pole_len · sin(θ),  cart_top − pole_len · cos(θ))
```

**Primitives (float coverage, then `_over` alpha composite)**

- `_stroke_line` — distance-to-segment coverage, line widths 2 (rail), 1 (cart edge), 3 (pole)
- `_fill_rect` — axis-aligned coverage for the cart body
- `_fill_circle` — pole tip radius `4.0` px
- Output: `np.clip(np.rint(img), 0, 255).astype(uint8)`

**Fixed colors**

| Element | RGB |
|---------|-----|
| Background `_BG` | `(245, 245, 245)` |
| Rail `_RAIL` | `(80, 80, 80)` |

**Appearance palettes** (`CART_APPEARANCE_PALETTE`, 8 entries, IC)

Fig. 5(c) shows the same physical state with different cart colors. Palettes are sampled **once per trajectory** by `sample_appearance(rng)` using the trajectory seed (`generate_trajectory` → `env.renderer.set_appearance`). Default (no sample) is palette 0: blue cart / red pole, used by centroid validation.

Palette 0: cart `(40,90,180)`, edge `(20,40,90)`, pole `(200,60,40)`, tip `(220,80,50)`. Remaining palettes swap cart/pole hues (red, green, purple, gold, teal, gray, navy).

## 1.7 Factory

`build_inverted_cartpole_env(config)` reads `simulation` and `simulation.physics` and constructs `CartPoleParams` + `InvertedCartPoleEnv`.

## 1.8 Environment validation (plan §3.3)

`validate_plan_environment(config)` runs after `assert_plan_environment_config`. Checks:

| Check | What it asserts |
|-------|-----------------|
| `custom_backend_not_gym` | class lives in `ts_jepa.env` |
| `force_limits` | clip `±50`/`±100`/`25` → `±20`; rollout commands in bounds |
| `state_integration` | finite states that change under force |
| `pendulum_dynamics` | `\|θ\|` grows from `0.06` rad under `u=0` over 150 steps |
| `cart_position_tracking` | `u=−15` from `x=0.35` reduces `\|x\|`; desired origin; score at origin = 1 |
| `rgb_rendering` | `uint8`, 3 channels, `[0,255]` |
| `native_render_resolution` | matches YAML camera `128×256×3` |
| `encoder_input_64x128` | eval preprocess yields `[3, 64, 128]` |
| `subpixel_force_motion` | 1 ms / 12 N changes RGB (MAPE > 0) |
| `numerical_stability_1ms` | 10 000 random-force steps stay finite |
| `state_render_consistency` | cart pixel centroid increases with `x` |
| `kappa_context_construction` | supervised concat `[6,64,128]`; JEPA frame `[3,64,128]`; eval deterministic |

`assert_plan_environment_config` additionally requires: environment name `inverted_cart_pole`, RGB, `τ_o=1 ms`, `dt=0.001`, 100 steps, `±20 N`, stride 1, `κ=2`, 3 channels, resize `[64,128]`.

---

# 2. Preprocessing

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/preprocessing/pipeline.py` | `PreprocessPipeline` |
| `src/ts_jepa/preprocessing/command_stats.py` | `CommandNormalizer` z-score |
| `src/ts_jepa/plan/preprocessing.py` | Paper values + order asserts |
| `src/ts_jepa/data/datasets.py` | `fit_command_normalizer` / `load_command_normalizer` |

**Config blocks:** `input.*`, `preprocessing.*`

## 2.1 RGB pipeline order

**Training** (`TRAINING_PIPELINE_STAGES`):

```text
uint8 HWC RGB
  → decode to float CHW in [0, 1]
  → color jitter (4 ops, random order)
  → color drop (p = 0.05)
  → ImageNet normalize
  → 5×5 Gaussian-kernel resample to 64×128   (σ ~ U[0.1, 0.2])
  → tensor [3, 64, 128]
```

**Evaluation** (`EVAL_PIPELINE_STAGES`):

```text
uint8 HWC RGB
  → decode to float CHW in [0, 1]
  → ImageNet normalize          (no jitter, no color drop)
  → 5×5 Gaussian resample to 64×128   (σ = 0.15, range midpoint)
  → tensor [3, 64, 128]
```

Eval uses the **same normalize-then-resize order** as training so ImageNet stats are applied in the same domain. Stochastic augmentations are off.

`process_frame(frame, stochastic=None, rng=None)`: `stochastic` defaults to `self.training`. If `rng` is omitted under training, a fresh `np.random.default_rng` is created from `np.random.randint`.

## 2.2 Decode

`_decode_frame`: contiguous `uint8 [H,W,3]` → `float32 [3,H,W] / 255.0`. Native stored size is `128×256`; encoder size is produced only by Gaussian resize.

## 2.3 Color jitter (paper values)

Each call shuffles `["brightness", "contrast", "saturation", "hue"]` and applies all four. Magnitudes:

| Op | Config | Implementation |
|----|--------|----------------|
| brightness | `0.05` | `img ← clamp(img · (1 + U[−0.05, 0.05]), 0, 1)` |
| contrast | `0.10` | `c = 1 + U[−0.10, 0.10]`; `mean` over H,W per channel; `clamp((img−mean)·c + mean)` |
| saturation | `0.10` | Rec.601 luma `0.299 R + 0.587 G + 0.114 B`; mix `img·s + gray·(1−s)` |
| hue | `0.05` | `hue_delta ~ U[−0.05, 0.05]`; skip if `|Δ| ≤ 1e-8` |

**Hue** (`_apply_hue`): YIQ rotation by `angle = hue_delta · 2π`:

```text
Y = 0.299 R + 0.587 G + 0.114 B
I = 0.596 R − 0.275 G − 0.321 B
Q = 0.212 R − 0.523 G + 0.311 B
I' = I cos α − Q sin α
Q' = I sin α + Q cos α
R' = Y + 0.956 I' + 0.621 Q'
G' = Y − 0.272 I' − 0.647 Q'
B' = Y − 1.106 I' + 1.703 Q'
```

Then clamp to `[0, 1]`.

## 2.4 Color drop (paper)

With probability `p = 0.05`, replace the RGB tensor by luma replicated on three channels (same Rec.601 weights). Otherwise identity.

## 2.5 ImageNet normalization (paper)

Per-channel, broadcast as `[3,1,1]`:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
out  = (img − mean) / std
```

Applied **before** resize, on `[0,1]` (or jittered) native-resolution tensors.

## 2.6 Gaussian-kernel resize (paper geometry, IC resampling)

Not blur-then-bilinear. Separable 5-tap Gaussian resampling in **input coordinates**.

- Kernel size: `gaussian_kernel = [5, 5]` → `n_taps = 5` per axis
- Training σ: `U[0.1, 0.2]`
- Eval σ: `0.5 · (0.1 + 0.2) = 0.15`
- Output: `resize = [64, 128]` = `(H, W)`

For output pixel centers:

```text
cy[i] = (i + 0.5) · (in_H / out_H) − 0.5     # i = 0..63,  in_H=128 → scale 2
cx[j] = (j + 0.5) · (in_W / out_W) − 0.5     # j = 0..127, in_W=256 → scale 2
```

1-D weights for taps `offs = arange(5) − 2`:

```text
loc  = center + offs
w    = exp(−(loc − center)² / (2σ² + 1e-12))
w    ← w / sum(w)          # per output pixel
idx  = round(loc).clamp(0, size−1)
```

Gather `img[:, iy, ix]` then `einsum("ciajb,ia,jb->cij", gathered, wy, wx)`.

Native `128×256` → encoder `64×128` is therefore a **2× downsample** with a 5×5 Gaussian kernel.

## 2.7 Context assembly

**JEPA / Algorithm 1** — one RGB frame:

- `assemble_jepa_frame(processed, t)` / `make_jepa_frame(frames, t)` → `[3, 64, 128]`
- `environment.jepa_observation: current_frame`
- `environment.context_tensor_shape: [3, 64, 128]`

**Supervised / autoencoder only** (`κ = 2`, IC channel concat):

- `assemble_context` / `make_context_tensor` packs `κ` consecutive processed frames ending at `t`
- Left-pad by repeating the first available frame if `t < κ−1`
- `torch.cat(..., dim=0)` → `[6, 64, 128]`
- `multi_frame_tensor_construction` must be `"channel_concat"`; anything else raises

`process_frames_cached(frames, start, end)` processes inclusive indices once and returns `{t: tensor}`.

## 2.8 Command z-score (paper; storage IC)

`CommandNormalizer` in `command_stats.py`:

```text
u_norm = (u − μ) / σ
u      = u_norm · σ + μ
```

- Fit on **JEPA train commands only** (`fit_command_normalizer`)
- `σ < 1e-8` is replaced by `1.0`
- Persisted at `{data_root}/stats/command_norm.json`
- Current `data_dp_fixed` stats: `μ ≈ 0.0292`, `σ ≈ 3.3047`

**Where z-score is used**

| Consumer | Domain |
|----------|--------|
| Predictor `Pφ` inputs | z-scored (`teacher_commands_norm`) |
| Semantic actor loss (Eq. 15) | **physical Newtons**, not z-score |
| Closed-loop `Pφ` on packet loss | z-score of last **clipped** plant force |

`assert_plan_preprocessing_config` checks jitter, drop p, ImageNet stats, kernel, σ range, resize, command_normalization `"z-score"`, and pipeline order lists.

---

# 3. Data

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/data/trajectory_generator.py` | Generate D_s / D_a `.npz` |
| `src/ts_jepa/data/datasets.py` | `TrajectoryDataset`, `ActorEmbeddingDataset` |
| `src/ts_jepa/data/temporal.py` | Indexing for κ / Kp windows |
| `src/ts_jepa/data/dataset_sanity.py` | Post-generation checks |
| `src/ts_jepa/plan/temporal.py` | `κ=2`, `Kp=15`, `emb=256` asserts |
| `scripts/pipeline/generate_trajectories.py` | CLI |

**Config blocks:** `dataset_generation.*`, `ts_jepa.dataset`, `semantic_actor.dataset`, `paths.data_root`

## 3.1 Dataset families

Paper: D_s for TS-JEPA, D_a for the semantic actor. Whether the underlying RGB rollouts are shared is **not specified**. This repo uses **disjoint `trajectory_index` / seed** so actor test is not a subset of JEPA train.

| Family | Purpose | Train | Test | Index range | Directory |
|--------|---------|-------|------|-------------|-----------|
| D_s | JEPA | 200 | 40 | `0 … 239` | `{data_root}/trajectories/jepa/{train,test}/` |
| D_a | Actor | 100 | 20 | `240 … 359` | `{data_root}/trajectories/actor/{train,test}/` |

`dataset_split_index_plan` concatenates counts in that order. `assert_plan_dataset_counts` requires exactly 200/40 and 100/20 and 100 steps, and that `dataset_generation` mirrors the same counts.

## 3.2 Trajectory generation loop

`generate_all_trajectories(config)`:

1. Assert dataset counts, environment config, preprocessing config
2. `build_env_and_teacher` (one env + one solved DP table for all rollouts)
3. For each split, `generate_dataset_split`
4. `sanity_check_trajectory_root(..., fit_normalizer=True)` — fail the run on `overall_pass=False`

Per trajectory (`generate_trajectory`):

```text
seed = 10_000 + trajectory_index
appearance = sample_appearance(default_rng(seed))   # per-trajectory palette
env.renderer.set_appearance(appearance)
env.rollout(teacher, steps=100, seed=seed)
```

Written as compressed `.npz` named `{trajectory_index:05d}.npz`.

**Stored arrays**

| Key | Dtype | Shape | Content |
|-----|-------|-------|---------|
| `frames` | uint8 | `[100, 128, 256, 3]` | Native RGB |
| `commands` | float32 | `[100]` | DP teacher force (N) |
| `states` | float64 | `[100, 4]` | Plant state at observe time |
| `split` | str | scalar | e.g. `jepa_train` |
| `seed` | int | scalar | `10000 + trajectory_index` |
| `trajectory_index` | int | scalar | global disjoint id |
| `control_teacher` | str | scalar | `"dp_nonlinear"` |
| `sampling_interval_ms` | float | scalar | `1.0` |
| `dt` | float | scalar | `0.001` |
| `observation_stride_steps` | int | scalar | `1` |
| `observation_interval_ms` | float | scalar | `1.0` |
| `render_height` / `render_width` | int | scalar | `128` / `256` |

Control source is **`dp_nonlinear`**, never `Uniform(-20, 20)`. `UniformRandomControlTeacher` exists only as an anti-pattern for tests.

`assert_native_frame_hw` rejects files already stored at `64×128` (old datasets). Those must be regenerated.

## 3.3 Temporal indexing (plan §6)

`κ = 2`, `Kp = 15`, embedding dim `256`.

For a sample at time index `k` on a length-`T=100` trajectory:

| Quantity | Indices | Length |
|----------|---------|--------|
| JEPA context frame | `{k}` | 1 RGB |
| Predictor commands | `u_k … u_{k+Kp−1}` | 15 |
| Target frames | `x_{k+1} … x_{k+Kp}` | 15 |
| Predicted latents | `z̃_{k+1} … z̃_{k+Kp}` | 15 |
| Horizon-NMAE commands | `u_{k+1} … u_{k+Kp}` | 15 |

Functions in `temporal.py`:

- `context_frame_indices(k, κ)` — `[k−κ+1, …, k]`, left-padded (supervised only)
- `target_end_indices(k, Kp)` — `[k+1, …, k+Kp]`
- `command_indices(k, Kp)` — `[k, …, k+Kp−1]`
- `predicted_command_indices(k, Kp)` — `[k+1, …, k+Kp]`
- `max_valid_time_index(T, Kp) = T − Kp − 1 = 84`

Valid JEPA starts: `k ∈ {0, …, 84}` → **85 windows / trajectory**.

## 3.4 `TrajectoryDataset` (JEPA)

Loads all `.npz` in a split into RAM. Builds `index_map: list[(file_idx, time_index)]`.

**`__getitem__` returns**

| Key | Shape | Notes |
|-----|-------|-------|
| `context` | `[3, 64, 128]` | `x_k` after preprocess |
| `future_frames` | `[15, 3, 64, 128]` | `x_{k+1:k+15}` |
| `teacher_commands` | `[15]` | physical N, `u_k:k+14` |
| `teacher_commands_norm` | `[15]` | z-scored, **Pφ conditioning** |
| `target_commands` | `[15]` | physical N, `u_{k+1:k+15}` |
| `target_commands_norm` | `[15]` | z-scored (horizon NMAE) |
| `time_index` | `()` long | `k` |

Training: stochastic preprocess on frames `k … k+Kp` only (cached per sample).  
Eval: every frame of every file is preprocessed once (`stochastic=False`) and reused.

`drop_last` on the train loader is true only if `len(train_set) ≥ effective_batch_size`.

**Sample counts (baseline, 200 train files, last 20 held out for val)**

| Split | Trajectories | Windows | Samples |
|-------|--------------|---------|---------|
| JEPA train (after holdout) | 180 | 85 | 15 300 |
| JEPA val (IC holdout) | 20 | 85 | 1 700 |
| JEPA test (untouched) | 40 | 85 | 3 400 |

Val files are the **last** `validation_trajectory_count=20` files in sorted train glob order (`_split_train_val`). Test is never used for selection.

## 3.5 `ActorEmbeddingDataset` (D_a)

Offline, once at actor-train start:

1. Load actor `.npz` (native RGB)
2. Eval preprocess every frame (no RGB augmentation — `training` arg is ignored)
3. Frozen `context_encoder` in `eval` + `no_grad`, microbatch 64
4. Store `(embedding [256], command_phys, command_norm)` for **every timestep** (no Kp window)

`__getitem__`:

| Key | Shape | Domain |
|-----|-------|--------|
| `embedding` | `[256]` | L2-normalized Ψθ output |
| `command` | `[1]` | physical Newtons (Eq. 15 target) |
| `command_norm` | `[1]` | z-score (stored, not used by MSE) |

| Split | Trajectories × steps | Samples |
|-------|----------------------|---------|
| Actor train full | 100 × 100 | 10 000 |
| Actor val (20% IC) | last 20% of train samples | 2 000 |
| Actor train after holdout | | 8 000 |
| Actor test | 20 × 100 | 2 000 |

Val split is by **sample index**, last `round(n · 0.2)` indices (`split_train_val_actor`), not by trajectory file.

## 3.6 Post-generation sanity

`sanity_check_trajectory_root`:

- File counts, length 100, finite values, native `128×256×3`, `control_teacher=dp_nonlinear`
- Train/test id disjoint within each family
- D_s vs D_a id disjoint
- Command std > 0, both signs present, ≥ 3 unique values
- `verify_not_uniform_random_actions`: histogram CV ≫ uniform-iid **or** lag-1 autocorr exceeds uniform-iid by `0.02`
- Command normalizer non-degenerate (`σ > 1e-8`)

## 3.7 Directory layout

```text
{data_root}/                          # e.g. data_dp_fixed/
  trajectories/
    jepa/train/00000.npz … 00199.npz
    jepa/test/00200.npz  … 00239.npz
    actor/train/00240.npz … 00339.npz
    actor/test/00340.npz  … 00359.npz
  stats/
    command_norm.json                 # {mean, std} from JEPA train
```

---

# 4. DP teacher

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/control/dp_teacher.py` | `DPGrid`, `DPControlTeacher` |
| `src/ts_jepa/data/trajectory_generator.py` | `build_env_and_teacher` |

**Config block:** `control_teacher.*`

Method is paper-specified (nonlinear DP). Numeric grids, `R`, horizon `K`, interpolation, and the local `act()` search are IC. `dp_substeps=1` is **paper-implied**: one Bellman transition = one `τ_o` physics step.

## 4.1 Discretization

**Force table** — `np.linspace(force_min, force_max, force_bins)`:

```text
u ∈ {−20, −16, −12, −8, −4, 0, +4, +8, +12, +16, +20}   # 11 bins
```

**State grid** (`DPGrid.from_config`, `np.linspace(lo, hi, n)`):

| Axis | Range | Nodes | Spacing |
|------|-------|-------|---------|
| `x` | `[−1.2, 1.2]` | 13 | 0.2 m |
| `ẋ` | `[−2.5, 2.5]` | 11 | 0.5 m/s |
| `θ` | `[−0.35, 0.35]` | 13 | ≈ 0.05833 rad |
| `θ̇` | `[−3.0, 3.0]` | 11 | 0.6 rad/s |

Total nodes: `13 × 11 × 13 × 11 = 20 449`.

Tables: `self.value[nx, nxd, nth, nthd]`, `self.policy` same shape (stores the discrete force chosen at that node). Meshgrid uses `indexing="ij"`.

## 4.2 Stage cost (IC numerics; one-step structure follows Eq. 3)

One DP decision holds `u` for `dp_substeps=1` physics tick (`_rollout_constant_force`):

```text
s'     = ODE.step(s, u, process_noise_std=0)
ℓ(s,u) = ½ ‖s' − s★‖²  +  ½ R u²
R      = control_effort_weight = 0.001
s★     = [0,0,0,0]
```

Control cost is charged **once per decision**, not per hypothetical substep. Discount `γ = 1.0` (Eq. 3 is undiscounted). `γ=0.99` per 1 ms step is rejected: that horizon is ~100 ms and restoring Q loses to `½ R u²`.

Finite horizon: `value_iteration_iters = 300` backups ≈ 300 ms ≈ `1.3 / ω_pendulum`.

## 4.3 Value continuation (IC): nodal Taylor

At 1 ms, `±20 N` moves `θ̇` by ~0.06 rad/s vs 0.6 rad/s grid spacing, so nearest-neighbor maps every force to the **same cell** and `u=0` wins on control cost. Multilinear interpolation still fails: convex `V` plus a velocity node at 0 creates a kink, so any 1 ms motion looks costly.

Continuation:

```text
s_g  = nearest grid node to s'
∇V   = np.gradient(V; edge_order=1)     # central difference on the table
V(s') ≈ V(s_g) + ∇V(s_g) · (s' − s_g)
```

Exact at nodes. Recovers the mixed partial `∂²V / ∂θ ∂θ̇` that prefers restoring forces at 1 ms. Implemented in `_interpolated_value`. Nearest indices via `searchsorted` + closer-of-left/right (`_quantize_indices`).

YAML `value_interpolation: nodal_taylor` documents this; the code path is the Taylor form (no runtime switch).

## 4.4 Value iteration (`_solve`)

1. Precompute, for each of 11 forces, the successor tensor and stage-cost tensor over all 20 449 nodes (vectorized `ode.step`).
2. For `K = 300` sweeps:
   - compute `∇V` once
   - for each force, `total = stage + γ · V_interp(s')`
   - take `argmin`; on ties (`|Δ| ≤ 1e-12`) prefer **smaller `|u|`**
3. Store `value`, `policy`.

Bellman cost used by both VI and `act()`:

```text
Q(s,u) = ℓ(s,u) + γ V(s')
```

## 4.5 Online action (`act`)

Not a raw table lookup. Local discrete re-optimization with the same one-step Bellman cost:

1. `table_u = policy[nearest node to s]`
2. Local candidates: `clip(table_u + {0, −4, +4, −8, +8}, −20, 20)`
3. Coarse table subset: `forces[:: max(1, len(forces)//5)]` = every 2nd bin  
   `{−20, −12, −4, +4, +12, +20}`
4. Unique union of local + coarse + table
5. Pick min `Q`; ties → smaller `|u|`

Returns a scalar in the 11-bin set (after clip). Env clips again to `±20`.

## 4.6 Constructor guards

`dp_substeps` must be `1` or `__init__` raises. `build_env_and_teacher` also asserts `dt == τ_o/1000` and `dp_substeps==1`.

`effective_dp_dt = dp_substeps · env.ode.dt = 0.001 s`.

---

# 5. Encoders: context, target, predictor

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/models/encoder.py` | `ResidualBlock`, `ContextEncoder`, alias `TargetEncoder` |
| `src/ts_jepa/models/ema.py` | Init copy, freeze, EMA |
| `src/ts_jepa/models/predictor.py` | `Predictor` MLP |
| `src/ts_jepa/models/ts_jepa.py` | `TSJEPA` wrapper |
| `src/ts_jepa/models/predictor_command_resolution.py` | OPEN ũ-source bookkeeping |
| `src/ts_jepa/plan/encoder.py` | Widths / EMA asserts |
| `src/ts_jepa/plan/predictor.py` | 1024 / 256 / AR asserts |

**Config:** `ts_jepa.encoder`, `ts_jepa.target_encoder`, `ts_jepa.predictor`, `ts_jepa.predictor_command_resolution`

`TSJEPA.__init__` calls `assert_plan_encoder_config`, `assert_plan_predictor_config`, `assert_plan_predictor_command_resolution`.

Parameter counts (baseline, counted from constructed modules):

| Module | Parameters |
|--------|------------|
| Context encoder Ψθ | 4 880 192 |
| Target encoder Ψθ̄ | 4 880 192 (frozen copy) |
| Predictor Pφ | 526 592 |
| JEPA trainable (θ + φ) | 5 406 784 |

## 5.1 Residual block (IC internals)

`ResidualBlock(in_channels, out_channels, stride=1)` — basic 2-layer residual:

```text
x
├─ Conv2d 3×3 s=stride p=1 bias=False → BN → ReLU
│  Conv2d 3×3 s=1      p=1 bias=False → BN
└─ skip: Identity  if stride=1 and in==out
         else Conv2d 1×1 s=stride bias=False → BN
add → ReLU
```

Paper-specified at the stage level: widths 64 → 128 → 256 with BN + ReLU. Kernel/stride/padding/block count are IC.

## 5.2 Context encoder Ψθ

Class `ContextEncoder`. Input: **one RGB frame** `[B, 3, 64, 128]` (Algorithm 1). `in_channels` comes from `input.channels_per_rgb_frame=3`, not `3κ`.

Constructor constraints:

- `widths` must be `(64, 128, 256)`
- `embedding_dim` must be `256` unless `strict_baseline_dim=False` (Fig. 7 grid via `experiments.allow_non_baseline_embedding_dim`)

**Stem (IC, not paper architecture)**

```text
Conv2d 7×7, stride 2, padding 3, 3→64, bias=False
BatchNorm2d(64)
ReLU(inplace)
MaxPool2d 3×3, stride 2, padding 1
```

`[B,3,64,128] → [B,64,16,32]`

**Stages** (`blocks_per_stage=2`, IC)

| Stage | Blocks | First stride | Tensor |
|-------|--------|--------------|--------|
| `stage64` | Res(64→64, s1), Res(64→64, s1) | 1 | `[B, 64, 16, 32]` |
| `stage128` | Res(64→128, s2), Res(128→128, s1) | 2 | `[B, 128, 8, 16]` |
| `stage256` | Res(128→256, s2), Res(256→256, s1) | 2 | `[B, 256, 4, 8]` |

**Head (IC; not GAP)**

```text
AdaptiveAvgPool2d((4, 8))     # already 4×8 at this input size
flatten → 256·4·8 = 8192
Linear(8192, 256)
F.normalize(..., dim=-1, eps=1e-8)    # L2 unit sphere
```

Output `z ∈ R^{256}` with `‖z‖₂ = 1`. L2-normalize is IC: Eq. (13) is cosine / direction-only; unbounded embeddings + SGD 0.2 make 15-step AR oscillate.

`architecture_summary()` records stem string, stages, head, `l2_normalize=True`.

## 5.3 Target encoder Ψθ̄

- **Same class:** `TargetEncoder = ContextEncoder`
- **Init:** `initialize_target_from_context` = `copy.deepcopy(context)` then `requires_grad_(False)` and `eval()`
- `assert_target_initialized_from_context` checks bitwise equal params and no grads
- **Never in the optimizer.** Train loop keeps `target_encoder.eval()` even while context/predictor are in `train()`
- **Forward:** `TSJEPA.encode_targets(future_frames)` is `@torch.no_grad()`
  - Input `[B, Kp, 3, H, W]` → reshape `[B·Kp, 3, H, W]`
  - Chunk size `256` frames (VRAM; does not change math)
  - Output `[B, Kp, 256]`, already L2-normalized

**EMA** (`ema_update`, η = `0.99`, after every optimizer step):

```text
θ̄ ← η θ̄ + (1 − η) θ
```

Float buffers (BN running mean/var) use the **same EMA**. Integer buffers (`num_batches_tracked`) are copied. `ema_step()` on `TSJEPA` applies this from context → target.

BN running stats of the **context** encoder still update on microbatches in train mode (no SyncBN / GhostBN). Target BN stats only move via EMA of those buffers.

## 5.4 Predictor Pφ

MLP, paper: hidden 1024, output 256, autoregressive, command-conditioned. Forbidden: virtual-channel / wireless features.

**Constructor constraints**

- `hidden_dim == 1024`
- `command_dim == 1` (scalar cart-pole force)
- `output_dim == embedding_dim`; both 256 unless Fig. 7 flag

**Layers (ReLU, concat, width 257, output L2 are IC)**

```text
x = concat(z, u_norm)          # [B, 257]
h = ReLU(Linear(257, 1024))    # fc_in, bias=True (nn.Linear default)
ẑ = Linear(1024, 256)          # fc_out
z̃ = F.normalize(ẑ, dim=-1, eps=1e-8)
```

Params: `257×1024+1024 + 1024×256+256 = 526 592`.

**One step** `forward_step(embedding [B,D], command_norm [B] or [B,1])` → `[B, D]` unit vector.

**Autoregressive unroll** `forward(embedding, commands_norm, horizon=None)`:

```text
commands_norm: [B, Kp] or [B, Kp, 1]
z_0 = z_k
for j in 0 .. Kp-1:
    z_{j+1} = normalize(Pφ(concat(z_j, u_{k+j})))
    store z_{j+1}
    z_j ← z_{j+1}.detach()     # IC: no 15-step BPTT
return stack → [B, Kp, D]
```

Detach is IC. Eq. (12) is still unrolled. Eq. (13) writes one-step cosine; full BPTT through 15 steps is unstable under Table II SGD 0.2. Encoder cosine therefore matches the **first** predicted step; Pφ is trained at every horizon step without backprop-through-time.

## 5.5 Predictor command source (plan §9, status OPEN)

Paper notation uses predicted commands `ũ` during predictor training; the pretraining generation mechanism is **not specified**. Status **must remain OPEN**; `paper_exact` must be `false`.

| Item | Current baseline |
|------|------------------|
| Selected candidate | `teacher_dp` |
| Training field | `batch["teacher_commands_norm"]` = DP `u*` z-scored |
| Implemented sources | `{teacher_dp}` only |
| Unimplemented candidates | `semantic_actor`, `sequential_actor_predictor`, `recovered_from_source` |
| Inference (packet lost) | last semantic-actor command (closed loop), **not** the training source |

`select_predictor_conditioning_commands` raises `NotImplementedError` for any non-`teacher_dp` source. Resolution is stored in JEPA checkpoints and `repetition_summary.json`.

## 5.6 `TSJEPA` methods

| Method | Behavior |
|--------|----------|
| `encode_context(x)` | Ψθ forward |
| `encode_targets(future, chunk_size=256)` | Ψθ̄, no grad, reshape Kp |
| `predict(z, u_norm, horizon=None)` | Pφ AR, default `self.kp=15` |
| `ema_step()` | EMA context → target |

Attributes: `command_source`, `command_resolution`, `ema_decay=0.99`, `kp=15`.

---

# 6. Training JEPA

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/training/jepa_procedure.py` | Algorithm 1 steps 1–6 |
| `src/ts_jepa/training/jepa_optimizer.py` | SGD groups, LR decay |
| `src/ts_jepa/training/train_jepa.py` | Loop, accumulation, ES, 5-seed protocol |
| `src/ts_jepa/losses/jepa_loss.py` | Cosine loss |
| `src/ts_jepa/plan/training.py` | Table II asserts |
| `src/ts_jepa/plan/procedure.py` | Algorithm 1 order asserts |
| `src/ts_jepa/plan/loss.py` | Cosine-only asserts |
| `src/ts_jepa/runtime.py` | DataLoader, prefetch, checkpoints, stall watchdog |
| `scripts/pipeline/train_jepa.py` | CLI |

**Config:** `ts_jepa.optimizer`, `ts_jepa.lr_decay`, `ts_jepa.loss`, `ts_jepa.training_procedure`, `ts_jepa.early_stopping`, `evaluation.*`, `runtime.*`

## 6.1 Algorithm 1 (one effective batch)

`jepa_forward_batch` (steps 1–4) then `jepa_sgd_and_ema_step` (steps 5–6):

```text
1. z_k     = Ψθ(x_k)                          # train mode, BN updates
2. z̄_{k+j} = Ψθ̄(x_{k+j})  j=1..Kp            # no_grad, eval
3. z̃       = Pφ(z_k, u_k:k+Kp−1)              # AR, detach between steps
4. L       = −mean_{b,j} cos_sim(z̃_{b,j}, z̄_{b,j})
5. clip ‖g‖₂ ≤ 1.0 (IC); SGD step on θ, φ only
6. EMA θ̄ ← 0.99 θ̄ + 0.01 θ; optimizer.zero_grad
```

Optimized modules: context encoder + predictor. Frozen: target encoder.

## 6.2 Loss

`jepa_cosine_similarity`: `F.cosine_similarity(pred, target.detach(), dim=-1, eps=1e-8)` → `[B, Kp]`.

`jepa_loss` = `-cos.mean()` over batch **and** horizon. Equivalent to mean over batch of `−(1/Kp) Σ_j cos_sim` for fixed Kp.

`cosine_alignment_loss` = `1 + jepa_loss` (same gradients; sometimes logged). Training uses `jepa_loss`.

Forbidden auxiliary losses: VICReg, reconstruction, MSE/L2, BYOL, Barlow Twins.

Because both Ψ and Pφ L2-normalize, `cos_sim = z̃ · z̄` (unit vectors), but the loss still goes through `F.cosine_similarity` with `eps=1e-8`.

## 6.3 Optimizer (Table II + IC)

`build_jepa_optimizer` → `torch.optim.SGD` only.

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Type | SGD | paper Table II |
| Learning rate | `0.2` | paper |
| Effective batch | `256` | paper |
| Epochs | `150` | paper |
| Weight decay | `0.0004` | paper |
| WD on BN affine | **0** | IC (`weight_decay_exclude_batchnorm`) |
| Momentum | `0.0` | IC (Table II silent) |
| Grad clip `‖g‖₂` | `1.0` | IC |
| EMA η | `0.99` | paper |
| LR decay | `×0.99` every 20 epochs | paper |
| Microbatch | `16` | IC (8 GB GPU) |
| Accumulation | `256/16 = 16` | IC |

`jepa_sgd_param_groups`: walk `context_encoder` and `predictor` modules; `BatchNorm1d/2d/SyncBatchNorm` parameters go to `weight_decay=0`; all other trainable params get `0.0004`. Target encoder is not in any group.

Forbidden draft defaults (asserted in `assert_plan_jepa_training_config`): Adam/AdamW, LR `0.001`, WD `1e-5`, 200 epochs, EMA `0.996`.

**LR schedule** (`should_apply_jepa_lr_decay`): after completing epoch `e` where `e > 0` and `e % 20 == 0`, multiply every param-group LR by `0.99`. Decays at epochs 20, 40, …, 140. After 150 epochs: `0.2 × 0.99^7`.

## 6.4 Microbatch accumulation (IC)

`resolve_jepa_batching`: `batch_size` must be divisible by `microbatch_size`. If micro > effective (tiny smokes), micro is clamped down.

`accumulation_plan(n_loader_batches, accum)`:

```text
n_effective = n_micro // accum
n_used      = n_effective * accum
leftover    = n_micro − n_used     # dropped; never optimizer.step()
```

Each micro backward uses `(loss / accum_steps).backward()` so accumulated grads match the mean over 256 samples. Incomplete trailing micros are `zero_grad`'d and discarded. Mismatch between planned and actual `optimizer.step()` count raises `RuntimeError`.

Train DataLoader batch size = **microbatch** (16), shuffle true, `drop_last` if enough samples. Eval loaders also use microbatch for VRAM; protocol (holdout / test) is unchanged.

## 6.5 Train loop (`train_ts_jepa`)

Per seed:

1. Seed numpy/torch/cuda with `seed`
2. `fit_command_normalizer` on JEPA train (rewrites `command_norm.json`)
3. Build `TrajectoryDataset` train (aug on) / test (aug off)
4. Carve last 20 train **files** as val
5. Construct `TSJEPA`, SGD, watchdog
6. For epoch `1 … 150` (or `max_epochs`):
   - `context_encoder.train()`, `predictor.train()`, `target_encoder.eval()`
   - CUDA prefetcher over `n_micros_used` micros
   - Algorithm 1 as above
   - Mean train loss over effective steps (one `.item()` per effective step, not per micro)
   - `evaluate_cosine_loss` on val (no SGD/EMA)
   - If val improves: save `best.pt` (CPU `state_dict`)
   - Else `stale += 1`
   - LR decay if due
   - Early stop if `stale >= patience` (`patience=20`, IC) and ES enabled
   - Save resumable `last.pt`
7. Reload best weights; evaluate **untouched** JEPA test; write `metrics.json`

**Early stopping / selection**

- Enabled (paper Section IV.A)
- Val trajectory count `20` is IC, from train only
- Selection criterion: **lowest validation cosine-alignment loss** (`jepa_loss`)
- Test loss is reported only

**Resume:** `last.pt` must contain `model`, `optimizer`, `epoch`, `best_val`, `best_epoch`, `stale`, `history`, `config`, `normalizer`. `validate_checkpoint_config_compatibility` compares Kp, κ, embedding, optimizer type/LR/WD/momentum, LR decay, batch/microbatch, EMA, encoder/predictor arch keys, command-resolution `selected_source` / `paper_exact`.

## 6.6 Five-seed protocol

`train_ts_jepa_repetitions`:

- Seeds `{0,1,2,3,4}` (`evaluation.seeds`), 5 repetitions, report **best**
- Skip a seed if `seed_training_complete`: `metrics.json` + `last.pt` with `test_loss` and epoch/history covering the requested epoch budget (a 2-epoch smoke must not skip a later 150-epoch run)
- Copy the seed with lowest `best_val` to `{runs}/best.pt`
- Write `repetition_summary.json` with criterion `best_validation_cosine_alignment_loss`

Run dir: `{runs_root}/{ts_jepa_dirname}/seed_{s}/` (overlay typically `runs/ts_jepa_dp_fixed/`).

## 6.7 Runtime plumbing (IC, not paper)

From `runtime.*` / `configure_training_runtime`:

| Item | Value |
|------|-------|
| DataLoader workers | 4 on CUDA, 0 on CPU |
| `pin_memory` | true (CUDA) |
| `persistent_workers` | true |
| `prefetch_factor` | 2 |
| `dataloader_timeout_s` | 120 |
| Heartbeat / stall | 30 s / 180 s → stdout + `train.log` |
| `cudnn.benchmark` | true (fixed 64×128) |
| TF32 | allowed on CUDA |
| CUDA memory fraction | 0.95 |
| Checkpoint I/O | CPU `state_dict` + sync before `torch.save` |
| Prefetch | `CUDAPrefetcher` overlaps H2D with compute |

Worker RNG: `np.random.seed(torch.initial_seed() % (2^31−1) + worker_id)`.

---

# 7. Semantic actor

**Source files**

| File | Role |
|------|------|
| `src/ts_jepa/models/actor.py` | `SemanticActor` |
| `src/ts_jepa/training/train_actor.py` | Train loop, freeze JEPA, 5-seed protocol |
| `src/ts_jepa/training/actor_helpers.py` | Val split, MSE eval, mean-command baseline |
| `src/ts_jepa/plan/actor.py` | Architecture + Table III asserts |
| `src/ts_jepa/inference/infer.py` | Closed-loop use of Cε |
| `scripts/pipeline/train_actor.py` | CLI |

**Config:** `semantic_actor.*`

## 7.1 Role

After TS-JEPA is trained, a **separate** network `C_ε` maps a frozen embedding to a control command:

```text
z_{i,k}  →  Cε  →  ũ_{i,k}   (scalar force, Newtons)
```

Input is **embedding only**. Desired state `x_d` is **not** concatenated (forbidden names: `desired_state`, `x_d`, `xd`, `state`, `rgb`, `frame`). During actor training the entire JEPA module is frozen and then deleted after embeddings are cached.

## 7.2 Architecture

Paper: two hidden layers 1024 and 256, ReLU, dropout 0.2 (Table III). Dropout **placement** and linear head are IC.

```text
z ∈ R^{256}
  Linear(256, 1024) → ReLU → Dropout(0.2)
  Linear(1024, 256) → ReLU → Dropout(0.2)
  Linear(256, 1)                # linear head, no tanh/sigmoid/clip
  → ũ ∈ R^{1}   physical Newtons
```

`nn.Sequential` exactly as above. `assert_plan_architecture` requires 3 `Linear` with shapes `(D,1024)`, `(1024,256)`, `(256,1)`, exactly 2 ReLUs, last module `Linear`, and rejects `Tanh`/`Sigmoid`/`Hardtanh`/`ReLU6`.

Parameter count: **525 825**.

`from_config` reads `semantic_actor.architecture.hidden_dims`, `dropout`, and `ts_jepa.encoder.embedding_dim`. Fig. 7 may change `D`; hidden dims stay 1024/256.

YAML `output_activation: linear`. Plant clip to `±20 N` happens at the env / `RuntimeCommandStats.apply_plant_force_limits`, not inside Cε.

## 7.3 Loss (Eq. 15)

```text
L_actor = MSE(ũ, u) = mean ‖u − ũ‖²
```

Domain: **physical Newtons** (`loss_domain: physical`). Z-score is TS-JEPA / Pφ only. Criterion: `nn.MSELoss()` (mean reduction). MAE/L1 is forbidden.

## 7.4 Optimizer (Table III + IC WD)

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Type | AdamW | paper Table III |
| LR | `0.006` | paper |
| Batch size | `200` | paper |
| Epochs | `300` | paper |
| Dropout | `0.2` | paper |
| Weight decay | `0.01` | IC (Table III silent on WD) |
| AdamW β / ε | PyTorch defaults `(0.9, 0.999)`, `1e-8` | IC |
| Early stopping | enabled | paper |
| Patience | `30` | IC |
| Val fraction | `0.2` of actor **train samples** | IC |

`_assert_optimizer_only_actor`: optimizer param ids must equal `actor.parameters()` exactly. `_assert_encoder_frozen`: after `requires_grad_(False)` on the loaded JEPA, no JEPA param may remain trainable.

## 7.5 Train loop (`train_semantic_actor`)

1. Load JEPA checkpoint (`--jepa-checkpoint` or `runs/.../best.pt`)
2. `TSJEPA.load_state_dict` → `eval()` → freeze all params
3. Load command normalizer from disk (do **not** refit)
4. `ActorEmbeddingDataset` on actor train/test using **context encoder only**
5. `del jepa` + CUDA cache empty
6. Split last 20% of train embeddings as val
7. `SemanticActor.from_config`; assert architecture when D=256 and hidden `[1024,256]`
8. For epoch `1 … 300`:
   - `actor.train()` (dropout on)
   - `pred = actor(emb)`; `loss = MSE(pred, command)` in Newtons
   - `zero_grad` → `backward` → `step` (no grad clip, no LR schedule)
   - Val MSE with `actor.eval()`
   - Save `best.pt` on val improvement; ES on `stale >= 30`
9. Reload best; untouched actor test MSE
10. Diagnostic IC: MSE of predicting the **train-mean command**; `beats_mean_command_baseline` if test MSE is lower. Not a paper metric / not a plan assert.

**Checkpoint payload (`best.pt`)**

```text
actor, jepa_checkpoint (resolved path), normalizer {mean,std},
config, val_loss, seed, selection_split="actor_train_holdout_validation"
```

`last.pt` also stores `test_loss`, `history`, mean-command baseline fields.

## 7.6 Five-seed protocol

`train_semantic_actor_repetitions`: same seed list `{0…4}`, skip completed seeds (`metrics.json` + `last.pt` with `test_loss` covering requested epochs). Select **lowest validation MSE**. Write `{runs}/best.pt` and `repetition_summary.json` with criterion `best_validation_mse`.

Typically trained against the selected 5-seed JEPA `best.pt`. Debug path: actor seed 0 against JEPA `seed_0/best.pt`.

## 7.7 Closed-loop use (plan §14, not a training module)

`FrozenRuntimeController` (`inference/infer.py`) — weights frozen:

| Event | Computation |
|-------|-------------|
| Packet received | eval-preprocess current RGB → Ψθ → `z` → Cε → Newtons → clip `±20` → z-score that force for Pφ |
| Packet lost | `z̃' = Pφ.forward_step(z, last_u_norm)` → Cε → clip → update `z`, `last_u_norm` |
| First loss with no latent yet | **`0.0` N** (IC) |

Device always observes RGB; transmission only gates whether the remote uses a fresh `z` or rolls Pφ. `plant_state` is ignored (embedding-only control).

---

## Source map (quick)

| Section | Primary implementation |
|---------|------------------------|
| Env ODE / env / render | `env/cartpole_ode.py`, `env/cartpole_rgb.py`, `env/renderer.py` |
| Preprocess RGB / commands | `preprocessing/pipeline.py`, `preprocessing/command_stats.py` |
| Data gen / datasets | `data/trajectory_generator.py`, `data/datasets.py`, `data/temporal.py` |
| DP teacher | `control/dp_teacher.py` |
| Ψθ / Ψθ̄ / Pφ | `models/encoder.py`, `models/ema.py`, `models/predictor.py`, `models/ts_jepa.py` |
| JEPA train | `training/jepa_procedure.py`, `training/jepa_optimizer.py`, `training/train_jepa.py`, `losses/jepa_loss.py` |
| Actor | `models/actor.py`, `training/train_actor.py` |

Config source of truth: `configs/ts_jepa_baseline.yaml`. Plan asserts: `src/ts_jepa/plan/*.py`.
