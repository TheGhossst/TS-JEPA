# Implementation Choices

This file documents every detail required to run the baseline that is **not**
explicitly paper-specified in `docs/TS-JEPA_Paper-Faithful_Final.md`.

Do not treat these as paper facts.

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
| Raw render size | `96 × 192` | not a paper-required camera resolution |

## DP teacher

| Choice | Value | Notes |
|--------|-------|-------|
| Method | discretized discounted value iteration + local action refinement | paper requires nonlinear DP; grids/`R` unspecified |
| Control effort weight `R` | `0.001` | scalar stand-in for positive-definite `R` |
| Discount | `0.99` | unspecified |
| Value-iteration iters | `25` | unspecified |
| Force bins | `11` in `[-20,20]` | unspecified discretization |
| State grids | see `configs/ts_jepa_baseline.yaml` | unspecified |

## Input tensor construction

| Choice | Value |
|--------|-------|
| `κ=2` frame packing | channel-concat → `[B, 6, H, W]` |
| Future target inputs | same κ construction ending at each future time |

## Encoder ResNet low-level structure

| Choice | Value |
|--------|-------|
| Stem | `3×3` conv → BN → ReLU |
| Blocks per stage | `2` |
| Downsampling | stride-2 first block of stages 128 and 256 |
| Pooling | adaptive average pool to `1×1` |
| Projection | linear to embedding dim |

## Predictor input construction

| Choice | Value |
|--------|-------|
| Per-step input | `concat(z, u_norm)` with `u_norm` shape `[B,1]` |
| Multi-step | autoregressive feedback of predicted `z` |
| JEPA training commands | trajectory/teacher DP control sequence (normalized) | not Semantic Actor `ũ` |

## Semantic actor output

| Choice | Value |
|--------|-------|
| Network output | linear (normalized command domain) |
| Runtime | `u = u_norm * σ + μ`, then clip to `[-20, +20]` N |

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
