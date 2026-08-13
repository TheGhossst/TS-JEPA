# Paper-Faithful Plan: Time-Series JEPA (TS-JEPA)

**Source (only):** Abanoub M. Girgis, Alvaro Valcarce, and Mehdi Bennis,
*Time-Series JEPA for Predictive Remote Control Under Capacity-Limited
Networks,* IEEE Internet of Things Journal, Vol. 13, No. 7, 1 April
2026. DOI: 10.1109/JIOT.2025.3650435.

This document is the implementation plan for reproducing **that paper**.
It must not deviate from the published text, equations, algorithms, and
tables.

### Evidence labels

-   **PAPER-SPECIFIED** — stated in the paper.
-   **PAPER-IMPLIED** — follows from equations/architecture, not written
    as a standalone sentence.
-   **NOT SPECIFIED** — the paper is silent. Do not invent a value and
    call it paper-exact.

A reproduction may still need executable choices where the paper is
silent. Those choices belong in `docs/IMPLEMENTATION_CHOICES.md`, not
here as paper requirements.

This paper does **not** contain Gilbert–Elliott burst channels, burst
masking, Burst Position Encoding, or a burst-aware JEPA objective. Do
not add them to this plan.

------------------------------------------------------------------------

# 1. What the paper proposes

A semantic-driven predictive control system plus channel-aware
scheduling for multiple inverted cart-pole devices under limited uplink
capacity.

Learned components (**PAPER-SPECIFIED**):

1.  Context encoder \(\Psi_\theta\) on the device: maps the
    high-dimensional state (RGB frame) to a low-dimensional embedding
    \(z_{i,k} = \Psi_\theta(x_{i,k})\).
2.  Target encoder \(\Psi_{\bar\theta}\) (training only): encodes future
    states; same architecture as the context encoder; stop-gradient;
    EMA update.
3.  Predictor \(P_\phi\) at the remote controller: predicts future
    embeddings from the current embedding and predicted control
    commands.
4.  Semantic actor \(C_\varepsilon\) at the remote controller: maps
    embeddings (received or predicted) to control commands.

Networking component (**PAPER-SPECIFIED**, not a neural block):

5.  Channel-aware scheduler: selects up to \(J\) devices using AoI,
    virtual queues, and required transmit power under InF-SH wireless
    conditions.

The paper’s reported headline results (abstract): 98.95% communication
cost reduction, normalized prediction error 0.004, 74.48% control
accuracy, robust 15-step prediction, and up to 16× more devices than
round-robin / opportunistic baselines with conventional control.

------------------------------------------------------------------------

# 2. Control system model

**PAPER-SPECIFIED** (Section II.A).

Devices \(i \in \mathcal{I}\). The sensor samples a \(p\)-dimensional
state at fixed sampling time \(\tau_o\):

``` text
x_{i,k} ∈ R^p     at t = k τ_o
```

The remote controller computes a \(q\)-dimensional command
\(u_{i,k} \in \mathbb{R}^q\), sent to the actuator over ideal downlink.

Dynamics (Eq. 1):

``` text
x_{i,k+1} = f_i(x_{i,k}, u_{i,k}) + n_{s,k}
```

\(n_{s,k}\) is i.i.d. Gaussian, zero mean, variance \(N_s\).

**NOT SPECIFIED:** a numerical cart-pole value of \(N_s\).

Target command (Eq. 2–3), solved by **dynamic programming**:

``` text
u*_{i,k} = arg min_{u_{i,k}} J(x_{i,k}, u_{i,k})

subject to Eq. (1) and u_min ≤ u_{i,k} ≤ u_max

J = (1/2) Σ_{k=0}^{K} ( ||x_{i,k} - x_d||_F^2 + u_{i,k}^T R u_{i,k} )
```

\(R\) is positive definite.

**NOT SPECIFIED:** numerical \(R\), cart/pole masses, pole length,
gravity, integrator, Gym vs custom backend, state-grid sizes.

Do **not** replace DP with LQR, or independent `Uniform(u_min, u_max)`
actions, and call that the paper policy.

**PAPER-IMPLIED:** the DP transition in Eq. (2) is the same discrete map
as Eq. (1), so one Bellman stage equals one \(\tau_o\) (1 ms in
Section IV). Do not hold \(u\) for a longer inner horizon and call that
the paper teacher.

------------------------------------------------------------------------

# 3. Simulation environment (inverted cart-pole)

**PAPER-SPECIFIED** (Section IV opening + control limits). The paper
uses “high-dimensional state” and “frame” interchangeably in
**Section II.C** (Problem Statement), not in the Section IV opening.

``` text
environment: inverted cart-pole
state: RGB frame
τ_o = 1 ms
trajectory length: 100 time steps
u_max = +20 N
u_min = −20 N
command: horizontal force on the cart
control policy: nonlinear policy solved by dynamic programming
```

The paper states that this 1 ms rate is selected to capture meaningful
temporal dynamics for predictive embedding learning, with further
analysis in Fig. 4.

Fig. 4 compares consecutive-frame MAPE **at different sampling rates**,
with and without augmentation. That figure is an analysis of temporal
diversity. It does **not** change the stated simulation sampling
interval. The main setup remains \(\tau_o = 1\) ms and 100 steps.

**NOT SPECIFIED:** native camera resolution before resize. Do not claim
64×128, 128×256, or any other native size as a paper camera
specification. The paper only specifies the **resize destination**
64×128 (Section IV.A).

Fig. 5(c) shows two frames with different cart colors and the same
physical state, to argue semantic invariance. That is an evaluation
illustration. The paper’s specified mechanism for spatial diversity is
the augmentation list in Section IV.A, not a mandatory per-trajectory
color sampler.

------------------------------------------------------------------------

# 4. Datasets

**PAPER-SPECIFIED** (Section IV.A):

``` text
D_s  (TS-JEPA):     200 train + 40 test trajectories
                    RGB frames paired with control commands
                    D_s = {x_{i,k}, u_{i,k}}_{k=1}^{K_s}

D_a  (semantic actor): 100 train + 20 test trajectories
                    embeddings paired with control commands
                    D_a = {z_{i,k}, u_{i,k}}_{k=1}^{K_a}
```

Each simulated trajectory is 100 time steps at \(\tau_o = 1\) ms.

The actor dataset is formed **after** TS-JEPA is trained, using the
context encoder to produce \(z_{i,k}\).

**NOT SPECIFIED:** validation-split size (early stopping is specified;
the count is not).

------------------------------------------------------------------------

# 5. Preprocessing

**PAPER-SPECIFIED** (Section IV.A). Training RGB pipeline, in the order
listed:

1.  **Color jittering:** brightness 0.05, contrast 0.1, saturation 0.1,
    hue 0.05, applied in random order for each patch.
2.  **Color dropping:** grayscale with probability 0.05, luma component.
3.  **Normalization:** per-channel mean `[0.485, 0.456, 0.406]`, std
    `[0.229, 0.224, 0.225]`.
4.  **Resizing:** frames are resized to **64 × 128** using a 5 × 5
    Gaussian kernel with standard deviation randomly sampled from
    `[0.1, 0.2]`.

Control commands: **z-score** normalization.

Testing: **resizing and normalization** only (no stochastic
augmentation), in the **same order as the numbered training list**
(normalize then Gaussian-resize) so train and eval share one tensor
domain.

**NOT SPECIFIED:** the paper’s test sentence does not order the two
stages. This repo follows the training numbered list for both splits.
Resize is **5×5 Gaussian-kernel resampling** to 64×128 (not
blur-then-bilinear).

------------------------------------------------------------------------

# 6. Consecutive frames \(\kappa\) and horizon \(K_p\)

**PAPER-SPECIFIED:**

-   Main comparison uses \(\kappa = 2\) consecutive frames
    (Section IV.D.3) for **supervised** \(\kappa\in\{2,4\}\) (and the
    generative AE). TS-JEPA itself follows Algorithm 1:
    \(z_{i,k}=\Psi_\theta(x_{i,k})\) on **one RGB frame**.
-   Abstract and evaluation use a **15-step** prediction horizon.

**PAPER-IMPLIED baseline embedding dimension:** 256, because the
predictor output layer has 256 neurons. Fig. 7 also treats embedding
dimension as a grid-search variable over candidates shown in that
figure: **64, 128, 256, 350**. The paper does not explicitly name which
candidate is the final deployed dimension.

Supervised / AE \(\kappa\) packing is **channel concat** to
`[B, 3\kappa, 64, 128]` (not written in the paper; IC for those
baselines only).

Formal encoder equation is \(z_{i,k} = \Psi_\theta(x_{i,k})\). Algorithm
1 encodes each future frame \(x_{i,k+j}\) with the target encoder,
\(j = 1 \ldots K_p\).

------------------------------------------------------------------------

# 7. Context encoder \(\Psi_\theta\)

**PAPER-SPECIFIED:** deep convolutional ResNet with layers of **64, 128,
and 256** neurons, each followed by batch normalization and ReLU.

**NOT SPECIFIED:** residual-block count, stem kernel/stride, padding,
pooling (including global average pool), and the exact head that maps
the last feature map to the embedding.

Do not write a stem `Conv 7×7`, MaxPool, GAP, or spatial `4×8` pool as
paper architecture.

------------------------------------------------------------------------

# 8. Target encoder \(\Psi_{\bar\theta}\)

**PAPER-SPECIFIED** (Section III.A, Eq. 14, Algorithm 1):

-   Same architecture as the context encoder.
-   Initialize \(\bar\theta \leftarrow \theta\).
-   Block gradients through the target branch.
-   EMA: \(\bar\theta \leftarrow \eta\,\bar\theta + (1-\eta)\,\theta\).
-   Table II: \(\eta = 0.99\).

------------------------------------------------------------------------

# 9. Predictor \(P_\phi\)

**PAPER-SPECIFIED** (Eq. 12, Algorithm 1, architecture paragraph):

``` text
(ẑ_{i,k+1}, …, ẑ_{i,k+K_p})
  = P_φ( z_{i,k}  |  ũ_{i,k}, …, ũ_{i,k+K_p−1} )
```

MLP: hidden layer **1024**, output layer **256**. Autoregressive
embedding prediction conditioned on predicted commands.

**NOT SPECIFIED:** concat vs other fusion; input width 257; virtual
channel features (do not add them).

### Predicted commands during TS-JEPA pretraining

**NOT SPECIFIED.** Dataset \(D_s\) contains ground-truth \(u_{i,k}\).
Eq. 12 and Algorithm 1 write \(\tilde u\). The semantic actor is
trained **after** TS-JEPA, so it cannot be the paper’s pretraining
\(\tilde u\) source.

Status remains **OPEN**. Do not claim teacher forcing is
paper-specified. Any executable stand-in must be labeled an
implementation choice.

------------------------------------------------------------------------

# 10. TS-JEPA loss and Algorithm 1

**PAPER-SPECIFIED:** cosine similarity between predicted embeddings and
target embeddings. Optimize \(\theta\) and \(\phi\) by gradient descent.
Then EMA-update \(\bar\theta\).

Algorithm 1:

1.  \(z_{i,k} \leftarrow \Psi_\theta(x_{i,k})\)
2.  For \(j = 1 \ldots K_p\): \(z_{i,k+j} \leftarrow \Psi_{\bar\theta}(x_{i,k+j})\)
3.  Predict as in Eq. (12)
4.  Cosine-similarity loss
5.  Gradient update of \(\theta,\phi\) as in Eq. (13)
6.  EMA update of \(\bar\theta\)

Do not add VICReg, reconstruction, or other losses and call them
TS-JEPA.

------------------------------------------------------------------------

# 11. TS-JEPA hyperparameters (Table II)

**PAPER-SPECIFIED** (Table II + surrounding text):

``` text
optimizer: SGD
learning rate: 0.2
batch size: 256
epochs: 150
weight decay: 0.0004
EMA decay η: 0.99
LR decay: multiply by 0.99 every 20 epochs
```

Do not substitute Adam / LR=0.001 / EMA=0.996 and call it this paper.

Early stopping on validation performance is stated in Section IV.A.
**NOT SPECIFIED:** patience and validation-set size.

Section IV.A, immediately after describing both Table II (TS-JEPA) and
Table III (actor) training, states that each experiment is repeated
five times and the best results are reported. That protocol applies to
**both** models, not only the actor.

Hardware used in the paper: NVIDIA Tesla V100-PCIE-16GB. Not a
reproduction requirement.

------------------------------------------------------------------------

# 12. Semantic actor \(C_\varepsilon\)

**PAPER-SPECIFIED** (Section III.B, Eq. 15, Table III):

Trained after TS-JEPA. The context encoder is no longer updated
(**PAPER-IMPLIED:** training of \(\Psi_\theta\) has ended; the paper
does not use the word “frozen”). \(D_a\) is built from its embeddings.
Minimize MSE **on physical commands** \(u\) (Newtons), Eq. 15:

``` text
(1/K_a) Σ_k || u_{i,k} − ũ_{i,k} ||_2^2
```

Z-score normalization is specified for **TS-JEPA** training stability
(Section IV.A), not as the actor loss domain.

MLP: two hidden layers **1024** and **256**, ReLU after each hidden
layer, scalar force output for cart-pole.

Table III:

``` text
optimizer: AdamW
learning rate: 0.006
batch size: 200
epochs: 300
dropout: 0.2
early stopping: yes
```

Five repeats and reporting the best result: same Section IV.A sentence
as in Section 11 (covers TS-JEPA and the actor).

**NOT SPECIFIED:** output activation / clipping to \([-20,20]\).
Desired state \(x_d\) is used in the **scoring function**, not as a
documented actor input.

------------------------------------------------------------------------

# 13. Wireless model and scheduler

**PAPER-SPECIFIED** (Section II.B, III.C, Algorithms 2, Eqs. 4–10,
16–25).

Scenario: Indoor Factory Sparse High BS (**InF-SH**). Rayleigh block
fading, constant over \(\tau_o\), independent across slots. Shadow
fading standard deviation **4.0** (path-loss text). Tested SNR
thresholds \(\gamma_{th} \in \{5, 10, 20\}\) dB.

LoS path loss (Eq. 4):

``` text
PL_LoS_dB = 31.84 + 21.5 log10(D^{3D}_i) + 19 log10(W_c)
```

LoS probability (Eq. 5); NLoS uses Eqs. 6–7.

SNR (Eq. 8); capacity \(R_{i,k} = W_i \log_2(1+\gamma_{i,k})\) (Eq. 9);
outage (Eq. 10).

### AoI (Eq. 17 / Algorithm 2)

``` text
β_{i,k+1} = 1            if device i is scheduled (α_{i,k} = 1)
β_{i,k+1} = 1 + β_{i,k}  otherwise
```

Initialize \(\beta_{i,0} = 1\). Do **not** replace this with increments
of \(\tau_o\) unless reproducing a different paper.

### Virtual queue (Eq. 18)

``` text
Q_{i,0} = 0
Q_{i,k+1} = max(Q_{i,k} − β_{i,th}, 0) + β_{i,k}
```

### Required power (Eq. 23) and index (Eq. 25)

If \(p^{req}_{i,k} > p_{max}\), device is infeasible, \(S_{i,k} = -\infty\).
Else compute drift-plus-penalty index \(S_{i,k}\). Schedule up to \(J\)
devices with the largest **positive** \(S_{i,k}\). Scheduled devices
transmit at \(p_{i,k} = p^{req}_{i,k}\).

### Table IV (geometry / channel only)

**PAPER-SPECIFIED** — Table IV has exactly these 11 rows:

``` text
hall size:              300 × 150 m²
room height:            6 m
BS height h_BS:         10.0 m
device height h_{i,R}:  1.5 m
carrier frequency W_c:  3.75 GHz
total bandwidth:        20 MHz
clutter height h_c:     3 m
clutter size D_clutter: 2.0 m
clutter density δ:      60%
2D distance D^{2D}_i:   50 m
noise power N_c:        −95 dB
```

The paper’s carrier-frequency symbol is **\(W_c\)**, not \(f_c\).

**NOT SPECIFIED** anywhere in the paper (including Table IV): \(I\),
\(J\), \(V\), \(\beta_{i,th}\), \(p_{max}\). Algorithm 2 needs them to
run; treat them as implementation choices. Do not hunt Table IV for
them.

The Lyapunov constant \(B\) in Eq. (22) is **explicitly omitted** by
the paper (“does not affect the system performance in Lyapunov
optimization”). It is not a lookup value.

The scheduler is not an input to the predictor.

------------------------------------------------------------------------

# 14. Inference (paper)

**PAPER-SPECIFIED:**

-   After training, context encoder is deployed on devices; predictor
    and actor on the remote/cloud side.
-   If an embedding is received: actor maps it to \(\tilde u_{i,k}\).
-   If not: predictor rolls latent state forward using predicted
    commands; actor maps predicted embeddings to commands.

**PAPER-IMPLIED:** weights are not updated at inference (the paper
never says “frozen”). If the actor is trained in the z-score command
domain, invert z-score before applying Newtons. The paper states
z-score for training stability, not a separate runtime equation.

------------------------------------------------------------------------

# 15. Evaluation metrics

**PAPER-SPECIFIED** (Section IV.B).

### Encoder quality

t-SNE of embeddings (Fig. 5). Similar states should cluster; embeddings
should be robust to superficial visual differences (Fig. 5(c)).

### Temporal/spatial consistency (Eq. 26)

MAPE between consecutive frames (also Fig. 4).

### Prediction accuracy (Eq. 27)

``` text
N^u_{i,K_p}
  = (1/K_p) Σ |ũ_{i,k} − u_{i,k}|
    / |max(u) − min(u)|
```

over the prediction steps in testing.

### Control performance (Eq. 28)

``` text
R_{i,k} = 1  if |x_{i,k} − x_d| ≤ 0.05  AND  |ϑ_{i,k}| ≤ 0.05
R_{i,k} = 0  otherwise
```

Control accuracy is the fraction of steps with \(R_{i,k}=1\).
Scalability plots use acceptable score in **[0.74, 1.0]**.

### Communication efficiency

Bits to transmit the state or the embedding. Exact raw-frame bit
accounting must match the paper’s communication-cost figures, not an
invented \(256 \times 32\) identity unless that is how those figures
were computed.

------------------------------------------------------------------------

# 16. Paper baselines (Section IV.C)

Control:

1.  Optimal nonlinear DP policy on the high-dimensional state.
2.  Supervised model: high-dimensional state → command (\(\kappa=2\)
    and \(\kappa=4\) in Fig. 6).
3.  Generative autoencoder: reconstruct state, then nonlinear control.

Scheduling (with conventional control):

4.  Round-robin; hold last command if unscheduled.
5.  Opportunistic (channel-based); hold last command if unscheduled.

------------------------------------------------------------------------

# 17. Reported experimental claims to reproduce

These are paper results, not extra methods:

-   No-prediction (fresh embedding every slot) vs 15-step prediction
    with a single initial transmission (Fig. 6).
-   Embedding-dimension grid (Fig. 7).
-   Training-set size (Fig. 8).
-   Target SNR 5/10/20 dB (Fig. 9).
-   Scalability vs round-robin / opportunistic (Figs. 10–11), including
    packet loss.

------------------------------------------------------------------------

# 18. Explicitly not specified (must not be labeled paper-exact)

The paper is silent on every item below. A reproduction still needs
executable values; those are **implementation choices (IC)**, recorded
in `docs/IMPLEMENTATION_CHOICES.md` and `configs/ts_jepa_baseline.yaml`.

Do **not**:

-   put these values in `PLAN_*` paper-spec dicts;
-   fail `assert_plan_*` solely because an IC field changed;
-   name tests, comments, or errors as if the paper specified them.

`assert_plan_*` may still require that an IC **key exists** (Algorithm 2
cannot run without \(J\)). Existence is not a paper number.

Working baseline values are listed so they stay documented. Changing a
number here does **not** make it paper-exact.

### Native RGB / renderer

**NOT SPECIFIED:** native camera size before resize; renderer;
anti-aliasing.

Paper specifies only the **resize destination** 64×128 (Section IV.A).

**IC (this repo):** native render `128 × 256 × 3`; coverage anti-aliased
float rasterizer (`src/ts_jepa/env/renderer.py`). Config:
`simulation.render_height/width`, `environment.render_resolution`.

### Cart-pole physics and DP numerics

**NOT SPECIFIED:** masses, pole length, gravity, \(N_s\), \(R\),
integrator, Gym vs custom, DP grids / bins / discount.

**PAPER-IMPLIED:** one DP Bellman transition is one Eq. (1) step, i.e.
one \(\tau_o\) (`dp_substeps=1` when `dt=τ_o`).

**IC (this repo):** custom `InvertedCartPoleEnv`; \(M=1.0\) kg,
\(m=0.1\) kg, \(l=0.5\) m, \(g=9.81\), track \(2.4\) m, semi-implicit
Euler, \(N_s=0\); DP `R=0.001`, discount \(0.99\), 11 force bins,
grids in `control_teacher.grid`. Method remains nonlinear DP (that
method **is** paper-specified).

### Frame stacking (\(\kappa\))

**NOT SPECIFIED:** how \(\kappa\) frames become a tensor.

**IC (this repo):** `channel_concat` → `[6, 64, 128]` for \(\kappa=2\).
Config: `input.multi_frame_tensor_construction`. \(\kappa=2\) itself is
paper-specified; the 6-channel packing is not.

### ResNet internals

**NOT SPECIFIED:** residual-block count, stem, padding, pooling
(including GAP), embedding head.

Paper specifies widths **64, 128, 256**, BN, and ReLU.

**IC (this repo):** `blocks_per_stage=2`; stem Conv \(7\times7\) stride 2
→ BN → ReLU → MaxPool \(3\times3\) stride 2; spatial pool `4×8` (not
GAP) → flatten → linear. Do not call this stem/head paper architecture.

### Predicted commands \(\tilde u\) during JEPA pretraining

**NOT SPECIFIED.** Status **OPEN** (plan §9). Dataset \(D_s\) has
ground-truth \(u_{i,k}\); the actor is trained after TS-JEPA.

**IC (this repo):** `teacher_dp` (trajectory \(u^*\));
`predictor_command_resolution.paper_exact: false`. Inference on a lost
packet uses the last semantic-actor command (closed loop; separate from
pretraining). Tests must reject `paper_exact: true`, not require it.

### Predictor concatenation width

**NOT SPECIFIED:** concat vs other fusion; input width 257.

Paper specifies MLP hidden **1024**, output **256**, command-conditioned.

**IC (this repo):** `concat(z, u)` → `Linear(257, 1024)`. Config:
`predictor.input_tensor_construction`.

### Actor output activation

**NOT SPECIFIED:** tanh / sigmoid / clip to \([-20,20]\) on \(C_\varepsilon\).

Paper specifies hidden ReLU and a scalar force output.

**IC (this repo):** linear head in normalized command space; denorm then
plant clip to \([-20,+20]\) N. Config:
`semantic_actor.architecture.output_activation: linear`. Linear is the
baseline IC, not a paper activation.

### Validation size / patience

**NOT SPECIFIED:** validation cardinality and patience. Early stopping
itself is paper-specified (Section IV.A).

**IC (this repo):** JEPA: 20 train trajectories held out, patience 20
epochs. Actor: 20% of actor **train** embeddings, patience 30. Test
splits (40 / 20) are never used for selection.

### Embedding bit width

**NOT SPECIFIED** as a float format. Plan §15: match the paper’s
communication-cost **figures**, do not invent \(256\times 32\) unless
that is how those figures were computed.

**IC / recovered accounting (this repo):** 8 bits per RGB channel and 8
bits per embedding value, because
\(1 - (256\times 8)/(64\times 128\times 3\times 8) \approx 98.95\%\).
That recovery is an accounting hypothesis, not a paper-stated dtype.
Constant names must not say the paper specified 8-bit embeddings.

### Scheduler scalars \(I, J, V, \beta_{th}, p_{max}\)

**NOT SPECIFIED** in Table IV or the body. Algorithm 2 needs them to
run.

**IC (this repo):** \(I=4\), \(J=2\), \(V=1.0\), \(\beta_{th}=5.0\),
\(p_{max}=0.2\) W. Config keys: `num_devices`,
`max_devices_scheduled_J`, `drift_plus_penalty_V`,
`aoi_threshold_beta_th`, `p_max_watt`.

### Lyapunov constant \(B\) (Eq. 22)

The paper **omits** \(B\) (“does not affect the system performance in
Lyapunov optimization”). It is not a lookup value and must not be
fitted or asserted as paper-exact. This repo omits \(B\) from the index
(`lyapunov_B_omitted`).

### Out of scope (not in this paper)

Anything named GE-JEPA, BPE, Gilbert–Elliott, burst masking, or
burst-aware loss.

------------------------------------------------------------------------

# 19. Reproduction order (paper system only)

``` text
Inverted cart-pole, τ_o = 1 ms, 100 steps, u ∈ [−20, 20] N
        ↓
Nonlinear DP teacher (Eqs. 2–3) → D_s
        ↓
Preprocessing as Section IV.A
        ↓
TS-JEPA: ResNet 64/128/256, EMA target, MLP predictor, cosine loss
        Table II training
        ↓
Trained Ψ_θ (no further encoder updates) → D_a → semantic actor
        (Eq. 15, Table III)
        ↓
Validate encoding (t-SNE), NMAE (Eq. 27), control score (Eq. 28),
        15-step prediction
        ↓
InF-SH channel + Algorithm 2 + Table IV geometry/channel constants
        (I, J, V, β_th, p_max are implementation choices)
        ↓
Figs. 6–11 protocol
```

------------------------------------------------------------------------

# 20. Paper-faithful checklist

## Environment

-   [ ] Inverted cart-pole, RGB state.
-   [ ] \(\tau_o = 1\) ms, 100 steps.
-   [ ] \(u \in [-20, +20]\) N, horizontal force.
-   [ ] Nonlinear DP teacher (not uniform random, not LQR-as-paper).

## Data

-   [ ] \(D_s\): 200 / 40 trajectories of (frame, command).
-   [ ] \(D_a\): 100 / 20 trajectories of (embedding, command).
-   [ ] Actor data from the trained context encoder (no further encoder
        updates).

## Preprocess

-   [ ] Jitter 0.05 / 0.1 / 0.1 / 0.05, random order.
-   [ ] Color drop \(p=0.05\), luma.
-   [ ] ImageNet mean/std as published.
-   [ ] Resize to 64×128 with 5×5 Gaussian, \(\sigma \in [0.1, 0.2]\).
-   [ ] Test: resize + normalize only.
-   [ ] Command z-score.

## TS-JEPA

-   [ ] \(\kappa=2\) in the main comparison.
-   [ ] \(K_p = 15\).
-   [ ] ResNet widths 64→128→256, BN, ReLU.
-   [ ] Target = copy, stop-grad, EMA \(\eta=0.99\).
-   [ ] Predictor MLP 1024→256, command-conditioned.
-   [ ] No virtual-channel predictor inputs.
-   [ ] Cosine embedding loss; Algorithm 1 order.
-   [ ] SGD, LR 0.2, batch 256, 150 epochs, wd 0.0004, LR×0.99 / 20
        epochs.
-   [ ] Five experimental repeats; report best (Section IV.A; both
        models).
-   [ ] \(\tilde u\) pretraining source documented as NOT SPECIFIED.

## Actor

-   [ ] 1024 and 256 hidden, ReLU, MSE.
-   [ ] AdamW, LR 0.006, batch 200, 300 epochs, dropout 0.2.
-   [ ] Early stopping; same five-repeat / best-reported protocol as
        TS-JEPA.
-   [ ] Encoder not updated during actor training.

## Metrics / wireless

-   [ ] NMAE Eq. (27); score Eq. (28) with 0.05 thresholds.
-   [ ] t-SNE; Fig. 4 MAPE analysis.
-   [ ] AoI Eq. (17); queue Eq. (18); Algorithm 2.
-   [ ] Table IV geometry/channel constants only (hall, heights,
        \(W_c\), bandwidth, clutter, 2D distance, \(N_c\)).
-   [ ] \(I\), \(J\), \(V\), \(\beta_{th}\), \(p_{max}\) documented as
        NOT SPECIFIED.
-   [ ] \(\gamma_{th} \in \{5,10,20\}\) dB.
