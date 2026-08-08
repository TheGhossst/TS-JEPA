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

## Semantic actor output

| Choice | Value |
|--------|-------|
| Network output | linear (normalized command domain) |
| Runtime | `u = u_norm * σ + μ`, then clip to `[-20, +20]` N |

## Validation / early stopping

| Choice | Value | Notes |
|--------|-------|-------|
| JEPA val trajectory count | `20` held out from the 200 train trajectories | paper enables early stopping but does not specify count |
| JEPA patience | `20` epochs | unspecified |
| Actor patience | `30` epochs | unspecified |
| Actor val | actor test split used for early stopping monitor | unspecified |

## Wireless / scheduler (networking layer)

Table IV geometry/radio parameters follow the published paper summary.
The following closed forms remain implementation choices:

| Choice | Value |
|--------|-------|
| Max scheduled devices `J` | `2` |
| `p_max` | `0.2` W |
| Drift-plus-penalty `V` | `1.0` |
| Virtual-queue arrival | `1.0` |
| Eval device count | `4` |
| Path-loss + clutter mapping | free-space × density attenuation |
| Small-scale fading | Rayleigh amplitude |
| Index `S` | `V * AoI + Q - p_req / p_max` |

## Out of scope until baseline validated

- GE-JEPA
- Gilbert-Elliott masking
- Burst Position Encoding
- Burst-aware / packet-loss-conditioned neural architecture changes
