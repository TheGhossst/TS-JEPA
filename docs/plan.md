# Paper-Faithful Implementation Plan: Time-Series JEPA (TS-JEPA)

## 0. Purpose

This document is the updated implementation specification for
reproducing:

**Time-Series JEPA for Predictive Remote Control Under Capacity-Limited
Networks**

The goal is to reproduce the TS-JEPA baseline as faithfully as possible
**before** adding the GE-JEPA extension.

This version incorporates the corrections made during the
architecture/code cross-check:

-   Custom inverted cart-pole with RGB observations.
-   DP/nonlinear teacher-generated actions rather than independent
    random actions.
-   Exact TS-JEPA architecture constraints.
-   Exact TS-JEPA training hyperparameters recovered from the paper.
-   Exact semantic actor architecture/training settings recovered from
    the current implementation notes.
-   Predictor receives the current/predicted latent state and predicted
    control commands, without invented virtual-channel inputs.
-   Target encoder uses stop-gradient + EMA.
-   The predicted-command generation mechanism during TS-JEPA training
    remains an explicit unresolved implementation ambiguity.
-   Wireless scheduling is implemented only after the TS-JEPA + actor
    baseline is validated.
-   GE-JEPA is added only after the TS-JEPA baseline is experimentally
    validated.

------------------------------------------------------------------------

# 1. Overall System

The original TS-JEPA system contains three main learned/system
components:

  -----------------------------------------------------------------------
  Component               Location                Function
  ----------------------- ----------------------- -----------------------
  Context Encoder `Ψθ`    Device                  Maps high-dimensional
                                                  RGB state/frame to a
                                                  256-D latent embedding

  Target Encoder `Ψθ̄`     Training only           Produces stable
                                                  future-state target
                                                  embeddings

  Predictor `Pϕ`          Remote controller       Autoregressively
                                                  predicts future latent
                                                  embeddings

  Semantic Actor `Cε`     Remote controller       Maps latent embeddings
                                                  to control commands

  Channel-Aware Scheduler Controller/base station Determines which
                                                  devices transmit based
                                                  on wireless/channel
                                                  state and AoI
  -----------------------------------------------------------------------

The central TS-JEPA idea is to predict future **semantic embeddings**
instead of reconstructing future RGB frames.

The context encoder produces:

``` text
z_i,k = Ψθ(x_i,k)
```

The target encoder produces:

``` text
z̄_i,k+j = Ψθ̄(x_i,k+j)
```

The predictor produces:

``` text
z̃_i,k+1, ..., z̃_i,k+Kp
```

conditioned on the current embedding and predicted control commands.

The JEPA objective aligns predicted embeddings with target embeddings
using cosine similarity.

------------------------------------------------------------------------

# 2. Experimental Principle

The implementation must follow this order:

``` text
Exact CartPole environment
        ↓
Exact teacher / dataset generation
        ↓
Exact TS-JEPA architecture
        ↓
Resolve predicted-command ambiguity
        ↓
Train TS-JEPA
        ↓
Train semantic actor
        ↓
Validate TS-JEPA baseline
        ↓
Implement wireless channel
        ↓
Implement paper scheduler
        ↓
Evaluate packet-loss / sparse-transmission behavior
        ↓
Gilbert-Elliott burst channel
        ↓
Burst masking
        ↓
Burst Position Encoding
        ↓
Burst-aware JEPA objective
        ↓
GE-JEPA evaluation
```

Do **not** add GE-JEPA components before the baseline is validated.

The purpose is to ensure that any later improvement or degradation can
be attributed to the GE-JEPA extension rather than to an incorrect
TS-JEPA reproduction.

------------------------------------------------------------------------

# 3. Environment: Custom Inverted Cart-Pole

## 3.1 Observation

The model operates on RGB observations rather than directly consuming
the four-dimensional physical state.

Target rendering resolution:

``` text
Height = 64
Width  = 128
Channels = 3
```

For `κ = 2`, two consecutive RGB frames may be concatenated as the
context input:

``` text
[B, 6, 64, 128]
```

The exact context construction must remain consistent throughout
training and evaluation.

## 3.2 Control

The cart is controlled by a horizontal force:

``` text
u_i,k ∈ R^1
```

with:

``` text
u_min = -20 N
u_max = +20 N
```

Sampling period:

``` text
τ_o = 1 ms
```

The environment should implement the paper's custom inverted-cart-pole
dynamics rather than replacing it with Gym `CartPole-v1`.

The physical dynamics are represented abstractly as:

``` text
x_i,k+1 = f_i(x_i,k, u_i,k) + n_s,k
```

where `n_s,k` represents the process-noise term when enabled by the
reproduction.

## 3.3 Environment validation

Before training, verify:

-   Force limits.
-   State integration.
-   Pendulum dynamics.
-   Desired cart-position tracking.
-   RGB rendering.
-   Rendering resolution `64 × 128`.
-   Numerical stability at `1 ms`.
-   Consistency between physical state and rendered image.

------------------------------------------------------------------------

# 4. Dataset Generation

Two datasets are required.

  Dataset   Purpose                    Training              Test   Steps / trajectory
  --------- ---------------- ------------------ ----------------- --------------------
  `D_s`     TS-JEPA            200 trajectories   40 trajectories                  100
  `D_a`     Semantic actor     100 trajectories   20 trajectories                  100

Sampling period:

``` text
τ_o = 1 ms
```

## 4.1 Teacher-generated control

The dataset should be generated using the nonlinear /
dynamic-programming teacher policy used for the reproduction.

The teacher produces:

``` text
u_k*
```

The implementation must explicitly verify that it is **not** generating
actions independently as:

``` text
Uniform(-20, 20)
```

This is an important correction to the earlier implementation plan.

The teacher-generated trajectory should provide the state/frame and
corresponding control command sequence.

Conceptually:

``` text
state/frame_k
      ↓
teacher / DP policy
      ↓
u*_k
      ↓
environment step
      ↓
state/frame_{k+1}
```

------------------------------------------------------------------------

# 5. Preprocessing

## 5.1 Training preprocessing

The image preprocessing pipeline is:

``` text
raw RGB frame
    ↓
augmentation
    ↓
normalization
    ↓
resize / formatting
    ↓
encoder
```

The implementation notes specify:

-   Color jitter:
    -   brightness = `0.05`
    -   contrast = `0.1`
    -   saturation = `0.1`
    -   hue = `0.05`
-   Color dropping / grayscale probability = `0.05`
-   ImageNet normalization:
    -   mean = `[0.485, 0.456, 0.406]`
    -   std = `[0.229, 0.224, 0.225]`
-   Gaussian blur:
    -   kernel = `5 × 5`
    -   sigma sampled from approximately `[0.1, 0.2]`
-   Final resolution = `64 × 128`

## 5.2 Test preprocessing

Do not apply stochastic training augmentations during testing.

Use:

``` text
resize + normalization
```

## 5.3 Control normalization

Control commands are z-score normalized using training-set statistics:

``` text
u_norm = (u - μ_u) / σ_u
```

When reporting or applying physical control, convert back using:

``` text
u = u_norm * σ_u + μ_u
```

------------------------------------------------------------------------

# 6. TS-JEPA Architecture

## 6.1 Context Encoder `Ψθ`

The context encoder is a convolutional ResNet-style encoder.

Required channel progression:

``` text
64 → 128 → 256
```

Target embedding dimension:

``` text
d_z = 256
```

Conceptual structure:

``` text
Input
[B, 3, 64, 128]
(or [B, 6, 64, 128] for κ=2)

        ↓

Conv2D
3 → 64
kernel = 7
stride = 2

        ↓

BatchNorm
ReLU
MaxPool

        ↓

Residual stage: 64

        ↓

Residual stage: 128
stride = 2

        ↓

Residual stage: 256
stride = 2

        ↓

Global Average Pooling

        ↓

Linear projection

        ↓

256-D embedding
```

The exact number of residual blocks per stage should not be claimed as a
paper fact unless recovered from the source.

If the paper does not explicitly specify it, document the chosen
backbone as an implementation choice.

------------------------------------------------------------------------

# 7. Target Encoder `Ψθ̄`

The target encoder has the same architecture as the context encoder.

Initialization:

``` text
θ̄ ← θ
```

The target encoder is not optimized through backpropagation.

Its parameters are updated using EMA:

``` text
θ̄ ← η θ̄ + (1 - η) θ
```

The target branch must use stop-gradient:

``` python
with torch.no_grad():
    z_bar = target_encoder(...)
```

This is a fundamental part of the TS-JEPA training mechanism.

The source paper explicitly states that the target encoder mirrors the
context encoder, is initialized identically, blocks gradients, and is
updated through EMA.

------------------------------------------------------------------------

# 8. Temporal Configuration

Use:

``` text
κ = 2
K_p = 15
embedding dimension = 256
```

where:

-   `κ` = number of consecutive context frames.
-   `K_p` = prediction horizon.

For a training sample:

``` text
Context:
[x_i,k-1, x_i,k]

Control sequence:
[u_i,k, u_i,k+1, ..., u_i,k+Kp-1]

Target frames:
[x_i,k+1, x_i,k+2, ..., x_i,k+Kp]
```

The predictor should generate:

``` text
[z̃_i,k+1, ..., z̃_i,k+Kp]
```

------------------------------------------------------------------------

# 9. Predictor `Pϕ`

## 9.1 Required structure

The predictor is an MLP with:

``` text
input
   ↓
Linear
   ↓
1024
   ↓
ReLU
   ↓
Linear
   ↓
256
```

The predictor is applied autoregressively.

At step `j`:

``` text
current latent
        +
predicted control
        ↓
predictor
        ↓
next latent
```

Then the predicted latent becomes the next autoregressive state.

## 9.2 Predictor input

The predictor should receive:

``` text
current embedding
+
predicted control command
```

It must **not** receive invented:

-   virtual inputs,
-   virtual channel variables,
-   virtual channel embeddings,
-   unspecified future channel features.

The previous implementation plan incorrectly introduced a virtual-input
term. That should be removed from the baseline reproduction.

Conceptually:

``` python
z_current = z_i_k

for j in range(K_p):
    z_next = predictor(
        concat([
            z_current,
            u_tilde_j
        ])
    )

    z_pred.append(z_next)
    z_current = z_next
```

The exact tensor/input dimensionality must follow the actual predictor
implementation once the command-generation mechanism is resolved.

------------------------------------------------------------------------

# 10. IMPORTANT: Predicted-Command Training Ambiguity

This is the main unresolved point in the paper-faithful implementation.

The paper defines the predictor in terms of **predicted commands**:

``` text
ũ_i,k, ..., ũ_i,k+Kp-1
```

and Algorithm 1 refers to predicted control commands.

However, the available paper text does not fully specify how the
complete predicted-command sequence is generated during TS-JEPA
pretraining.

Therefore:

> **Do not silently invent a mechanism and label it as paper-exact.**

Possible implementation choices must be evaluated and documented
separately.

Candidate approaches include:

1.  Teacher / ground-truth control sequence used as the conditioning
    sequence.
2.  A separately trained semantic actor generates the command sequence.
3.  A sequential actor-predictor procedure.
4.  Another mechanism explicitly recovered from the original
    implementation/source.

The selected approach must be recorded in the experiment configuration
and paper/reproduction notes.

Until this is resolved, the predictor training implementation should be
considered:

``` text
PENDING PAPER/IMPLEMENTATION RESOLUTION
```

This is preferable to introducing unsupported virtual inputs.

------------------------------------------------------------------------

# 11. JEPA Loss

The TS-JEPA objective is cosine similarity between predicted and target
embeddings.

For each horizon step:

``` text
cos_sim(z̃_j, z̄_j)
=
(z̃_j · z̄_j)
/
(||z̃_j||₂ ||z̄_j||₂)
```

Loss:

``` text
L_JEPA
=
-(1/K_p)
Σ_{j=1}^{K_p}
cos_sim(z̃_j, z̄_j)
```

Target embeddings are generated using the EMA target encoder.

Example:

``` python
z_context = context_encoder(context)

with torch.no_grad():
    z_target = [
        target_encoder(frame)
        for frame in future_frames
    ]

z_pred = []
z_current = z_context

for j in range(K_p):
    z_next = predictor(...)
    z_pred.append(z_next)
    z_current = z_next

loss = -mean(
    cosine_similarity(z_pred[j], z_target[j])
    for j in range(K_p)
)
```

------------------------------------------------------------------------

# 12. Exact TS-JEPA Training Hyperparameters

Use the following values for the paper-faithful baseline:

  Parameter                                               Value
  --------------------- ---------------------------------------
  Optimizer                                                 SGD
  Learning rate                                           `0.2`
  Batch size                                              `256`
  Epochs                                                  `150`
  Weight decay                                         `0.0004`
  EMA decay                                              `0.99`
  LR schedule             multiply LR by `0.99` every 20 epochs
  Prediction horizon                                       `15`
  Consecutive frames                                        `2`
  Embedding dimension                                     `256`

Do not replace these with generic BYOL/JEPA defaults such as:

``` text
Adam
LR = 0.001
EMA = 0.996
weight decay = 1e-5
epochs = 200
```

Those values were present in an earlier implementation draft but should
**not** be used for the paper-faithful reproduction.

------------------------------------------------------------------------

# 13. TS-JEPA Training Procedure

For every training batch:

### Step 1 --- Context encoding

``` text
z_i,k = Ψθ(x_i,k)
```

### Step 2 --- Target encoding

``` text
z̄_i,k+j = Ψθ̄(x_i,k+j)
```

with gradients disabled.

### Step 3 --- Autoregressive prediction

Generate:

``` text
z̃_i,k+1
...
z̃_i,k+Kp
```

using the selected predicted-command mechanism.

### Step 4 --- JEPA loss

``` text
L_JEPA = -mean(cosine similarity)
```

### Step 5 --- Gradient update

Update:

``` text
θ
ϕ
```

using SGD.

### Step 6 --- EMA update

Update:

``` text
θ̄ ← η θ̄ + (1-η) θ
```

------------------------------------------------------------------------

# 14. Semantic Actor `Cε`

The semantic actor maps the 256-D embedding to the scalar cart control
command.

Architecture:

``` text
256
 ↓
Linear → 1024
 ↓
ReLU
 ↓
Linear → 256
 ↓
ReLU
 ↓
Linear → 1
```

Output:

``` text
ũ_i,k
```

The actor is trained as supervised regression.

Loss:

``` text
L_actor
=
MSE(ũ_i,k, u_i,k)
```

The TS-JEPA encoder is frozen during actor training.

------------------------------------------------------------------------

# 15. Semantic Actor Training

Use the trained TS-JEPA context encoder.

For every sample:

``` text
x_i,k
   ↓
frozen Ψθ
   ↓
z_i,k
   ↓
Cε
   ↓
ũ_i,k
```

Compute:

``` text
MSE(ũ_i,k, u_i,k)
```

and optimize only `Cε`.

The current implementation specification uses:

  Parameter                              Value
  ------------------- ------------------------
  Architecture          `256 → 1024 → 256 → 1`
  Hidden activation                       ReLU
  Dropout                                `0.2`
  Optimizer                              AdamW
  Learning rate                        `0.006`
  Batch size                             `200`
  Epochs                                 `300`
  Early stopping                       Enabled
  Repetitions                              `5`
  Selection             Best validation result

The five-repetition procedure should report the best result, consistent
with the implementation notes.

------------------------------------------------------------------------

# 16. Baseline Validation --- Mandatory Before Wireless

Before implementing the wireless scheduler, validate the TS-JEPA +
semantic actor system.

Required checks:

### 16.1 Embedding quality

Use t-SNE or another appropriate visualization to inspect whether
semantically similar states have meaningful latent structure.

### 16.2 Actor prediction

Evaluate:

``` text
NMAE
```

between predicted and ground-truth control.

### 16.3 Control performance

Run the closed-loop cart-pole system and calculate the control score.

### 16.4 Horizon prediction

Inspect prediction quality over:

``` text
1 ... 15
```

steps.

Do not only report the average loss.

### 16.5 Communication reduction

Compare latent transmission with raw RGB-frame transmission.

### 16.6 Stability

Check whether the system remains stable when using the selected
predicted-command mechanism and temporal resolution.

Only after these tests pass should the wireless scheduler be introduced.

------------------------------------------------------------------------

# 17. Wireless Channel Model

The wireless component should be implemented after the baseline.

The implementation sequence is:

``` text
InF-SH environment
        ↓
LoS / NLoS path loss
        ↓
Rayleigh block fading
        ↓
Required transmit power
        ↓
Feasibility test
        ↓
AoI update
        ↓
Virtual queue update
        ↓
Drift-plus-penalty score
        ↓
Select up to J devices
```

The paper uses an Indoor Factory Sparse clutter (InF-SH) scenario.

## 17.1 Channel capacity

Conceptually:

``` text
C_i,k = B log2(1 + γ_i,k)
```

## 17.2 Outage

``` text
P_out,i = P(γ_i,k < γ_th)
```

The tested SNR thresholds are:

``` text
γ_th ∈ {5, 10, 20} dB
```

## 17.3 Rayleigh block fading

Use a block-fading Rayleigh channel coefficient per transmission.

The exact wireless parameter values from Table IV must be recovered from
the paper before claiming a paper-exact wireless reproduction.

Do not silently substitute arbitrary values for:

-   bandwidth `B`
-   carrier frequency `f_c`
-   noise power `σ²`
-   path-loss exponent `α`
-   number of devices `I`
-   resource blocks `J`
-   AoI threshold `β_th`
-   Lyapunov parameter `V`

------------------------------------------------------------------------

# 18. AoI and Virtual Queue

The AoI logic must follow the paper.

At each time step:

### Successful transmission

``` text
Δ_i,k+1 = τ_o
```

### Unscheduled device

``` text
Δ_i,k+1 = Δ_i,k + τ_o
```

### Scheduled but failed/outage transmission

``` text
Δ_i,k+1 = Δ_i,k + τ_o
```

The virtual queue tracks the long-term AoI constraint.

Initialize:

``` text
Q_i,0 = 0
```

The implementation notes use:

``` text
Q_i,k+1 =
max(Q_i,k + β_i,th - τ_i,k, 0)
```

where `τ_i,k` is the relevant time-since-success quantity.

The queue must be updated consistently with the AoI state used in the
scheduler.

------------------------------------------------------------------------

# 19. Required Transmit Power

For a transmission requiring `N_b` bits:

``` text
N_b = 256 × 32
    = 8192 bits
```

The required power is computed from the channel reliability/capacity
constraint.

Do not hard-code a simplified power formula until the paper's exact
notation and parameterization have been checked.

------------------------------------------------------------------------

# 20. Drift-Plus-Penalty Scheduler

The scheduler should:

1.  Observe channel state.
2.  Compute required transmission power.
3.  Mark infeasible devices.
4.  Compute the drift-plus-penalty scheduling score.
5.  Ignore non-positive candidates.
6.  Select at most `J` devices.
7.  Transmit selected devices.
8.  Update AoI.
9.  Update virtual queues.
10. Use predicted control for devices whose state is not freshly
    transmitted.

The exact values of:

``` text
I
J
V
β_th
p_max
```

must be recovered from the paper before final paper-exact experiments.

------------------------------------------------------------------------

# 21. Full Inference Loop

## 21.1 Device

``` python
def device_step(frame):
    frame = preprocess(frame)

    z = context_encoder(frame)

    if scheduled:
        transmit(z)
```

The transmitted embedding contains:

``` text
256 float32 values
= 8192 bits
```

## 21.2 Controller

If a fresh embedding is received:

``` text
current_z = received_z
```

Otherwise:

``` text
current_z = predictor(previous_z, predicted_command)
```

Then:

``` text
ũ = semantic_actor(current_z)
```

Convert the normalized command back to physical units and apply it to
the cart-pole controller.

For future unscheduled steps, use autoregressive latent prediction and
the semantic actor to generate future commands.

------------------------------------------------------------------------

# 22. Evaluation Metrics

## 22.1 Prediction NMAE

The normalized mean absolute error is:

``` text
NMAE =
(1/Kp)
Σ |ũ_k - u_k|
/
(max(u) - min(u))
```

For the cart-pole control range:

``` text
max(u) - min(u) = 40 N
```

Lower is better.

## 22.2 Control accuracy

Define:

``` text
R_i,k = 1
```

when:

``` text
|x_i,k - x_d| ≤ 0.05
AND
|θ_i,k| ≤ 0.05
```

Otherwise:

``` text
R_i,k = 0
```

Control accuracy is the fraction of evaluated time steps satisfying this
condition.

## 22.3 Temporal consistency

The implementation notes include a frame-level MAPE-style temporal
consistency metric.

Use the paper's exact definition when reproducing the reported value.

## 22.4 Communication efficiency

Embedding transmission:

``` text
256 floats × 32 bits
= 8192 bits
```

Compare this against the raw RGB-frame transmission baseline.

The paper reports a very large communication reduction; reproduce the
calculation using the exact frame representation and transmission
assumptions used in the paper rather than relying only on the nominal
embedding size.

## 22.5 Latent visualization

Use t-SNE as an auxiliary diagnostic to inspect the structure of learned
embeddings.

------------------------------------------------------------------------

# 23. Paper-Faithful Reproduction Checklist

## Environment

-   [ ] Custom inverted cart-pole implemented.
-   [ ] Paper dynamics verified.
-   [ ] Horizontal force control implemented.
-   [ ] `u ∈ [-20,20] N`.
-   [ ] `τ_o = 1 ms`.
-   [ ] RGB rendering implemented.
-   [ ] `64 × 128` resolution verified.

## Dataset

-   [ ] `D_s`: 200 train + 40 test trajectories.
-   [ ] `D_a`: 100 train + 20 test trajectories.
-   [ ] 100 steps per trajectory.
-   [ ] DP/nonlinear teacher implemented.
-   [ ] No independent uniform action sampling.
-   [ ] Train/test split verified.

## TS-JEPA

-   [ ] `κ = 2`.
-   [ ] `K_p = 15`.
-   [ ] Embedding dimension = 256.
-   [ ] Encoder channels = `64 → 128 → 256`.
-   [ ] Target encoder initialized from context encoder.
-   [ ] Target branch uses stop-gradient.
-   [ ] EMA target update implemented.
-   [ ] Predictor = `1024 → 256`.
-   [ ] No invented virtual-channel inputs.
-   [ ] Cosine embedding loss implemented.
-   [ ] Predicted-command mechanism explicitly documented.

## Training

-   [ ] SGD.
-   [ ] LR = 0.2.
-   [ ] Batch size = 256.
-   [ ] Epochs = 150.
-   [ ] Weight decay = 0.0004.
-   [ ] EMA = 0.99.
-   [ ] LR ×0.99 every 20 epochs.

## Actor

-   [ ] `256 → 1024 → 256 → 1`.
-   [ ] ReLU hidden layers.
-   [ ] Dropout = 0.2.
-   [ ] AdamW.
-   [ ] LR = 0.006.
-   [ ] Batch size = 200.
-   [ ] 300 epochs.
-   [ ] Early stopping.
-   [ ] Five repetitions.
-   [ ] Frozen TS-JEPA encoder.

## Baseline validation

-   [ ] t-SNE.
-   [ ] NMAE.
-   [ ] Control score.
-   [ ] 15-step latent prediction analysis.
-   [ ] Communication analysis.
-   [ ] Closed-loop stability.

## Wireless

-   [ ] InF-SH environment.
-   [ ] LoS/NLoS path loss.
-   [ ] Rayleigh block fading.
-   [ ] Required transmit power.
-   [ ] Feasibility check.
-   [ ] AoI update.
-   [ ] Virtual queue.
-   [ ] Drift-plus-penalty score.
-   [ ] Top-J scheduling.
-   [ ] Exact Table IV values recovered.

------------------------------------------------------------------------

# 24. GE-JEPA Extension

Only after the TS-JEPA baseline is validated:

``` text
Validated TS-JEPA
        ↓
Gilbert-Elliott channel
        ↓
Burst-loss masking
        ↓
Burst Position Encoding (BPE)
        ↓
Burst-aware JEPA objective
        ↓
GE-JEPA evaluation
```

## 24.1 What must remain unchanged

The extension should preserve:

-   Context encoder.
-   Target encoder.
-   EMA target update.
-   Stop-gradient target branch.
-   Basic predictor structure.
-   Future embedding prediction.
-   Base semantic actor unless a specific experiment evaluates a changed
    actor.

## 24.2 What GE-JEPA adds

GE-JEPA adds:

1.  Gilbert-Elliott channel model.
2.  Burst-loss masking.
3.  Burst Position Encoding.
4.  Burst-aware predictive loss.

The first packet-loss experiments should use the original validated
TS-JEPA predictor.

Do not simultaneously change the predictor, encoder, channel model, and
loss when first introducing burst losses.

------------------------------------------------------------------------

# 25. Experimental Ablation Plan

Recommended progression:

  Experiment   Description
  ------------ ----------------------------------------------
  E0           Original TS-JEPA, no packet loss
  E1           TS-JEPA + independent packet-loss simulation
  E2           TS-JEPA + wireless scheduling
  E3           TS-JEPA + Gilbert-Elliott burst channel
  E4           E3 + Burst Position Encoding
  E5           E4 + burst-aware JEPA objective
  E6           Full GE-JEPA

For every experiment report at minimum:

-   Prediction NMAE.
-   Control accuracy.
-   Closed-loop stability.
-   Communication cost.
-   Performance versus packet/burst loss.
-   Performance versus burst length.
-   Performance versus channel conditions.

This structure isolates the contribution of each GE-JEPA component.

------------------------------------------------------------------------

# 26. Known Unresolved Items

The following must not be silently treated as paper facts until
verified.

## 26.1 Predicted command generation during TS-JEPA training

The paper requires predicted commands but does not completely specify
how the entire command sequence is generated during training.

**Status: OPEN**

This is currently the most important reproduction ambiguity.

## 26.2 Exact residual-block counts

The text establishes the `64 → 128 → 256` channel progression, but the
exact number of residual blocks per stage should be verified before
calling a specific ResNet depth paper-exact.

**Status: VERIFY**

## 26.3 Exact Table IV wireless parameters

Recover:

``` text
B
f_c
σ²
α
I
J
β_th
V
p_max
```

from the paper's table before final wireless experiments.

**Status: VERIFY**

## 26.4 Exact communication baseline representation

The 8192-bit embedding size is straightforward, but the exact raw-frame
transmission calculation must use the same representation/assumptions as
the paper.

**Status: VERIFY**

------------------------------------------------------------------------

# 27. Reproduction Rules

The following rules should be followed throughout implementation:

1.  **Never replace an unspecified paper detail with a generic default
    and call it paper-exact.**
2.  **Keep paper facts and implementation choices explicitly
    separated.**
3.  **Do not add virtual inputs to the predictor unless the source
    explicitly requires them.**
4.  **Do not use random Uniform(-20,20) actions for the teacher dataset
    when reproducing the current corrected setup.**
5.  **Validate TS-JEPA before implementing wireless scheduling.**
6.  **Validate the wireless baseline before introducing Gilbert-Elliott
    burst losses.**
7.  **Introduce BPE and the burst-aware loss as separate ablations.**
8.  **Keep the original TS-JEPA architecture fixed while measuring the
    contribution of GE-JEPA components.**
9.  **Record every unresolved implementation choice in experiment
    configuration files and experiment logs.**
10. **Do not report an implementation choice as an original-paper
    fact.**

------------------------------------------------------------------------

# 28. Final Implementation Roadmap

``` text
PHASE 1
Custom inverted CartPole
        ↓
Validate dynamics + RGB rendering

PHASE 2
DP/nonlinear teacher
        ↓
Generate Ds / Da
        ↓
Validate trajectories and action distribution

PHASE 3
TS-JEPA encoder
        ↓
Target encoder
        ↓
Predictor
        ↓
EMA
        ↓
Cosine loss

PHASE 4
Resolve predicted-command generation
        ↓
Document decision

PHASE 5
Train TS-JEPA
        ↓
Validate latent prediction
        ↓
Validate embeddings

PHASE 6
Train semantic actor
        ↓
NMAE
        ↓
Closed-loop control score

PHASE 7
Freeze validated baseline
        ↓
Implement wireless channel
        ↓
Implement scheduler
        ↓
Validate packet-loss operation

PHASE 8
Gilbert-Elliott burst channel
        ↓
Burst masking

PHASE 9
Burst Position Encoding

PHASE 10
Burst-aware JEPA loss

PHASE 11
Full GE-JEPA evaluation
        ↓
Ablation study
        ↓
Paper results
```

------------------------------------------------------------------------

# 29. Baseline Definition

For the project, the canonical baseline should be:

``` text
Custom inverted CartPole
+
DP/nonlinear teacher
+
RGB observations
+
κ = 2
+
ResNet-style 64→128→256 encoder
+
256-D embedding
+
EMA target encoder
+
1024→256 predictor
+
Kp = 15
+
cosine JEPA loss
+
paper-faithful SGD training
+
frozen semantic actor
+
paper wireless scheduler
```

GE-JEPA is then defined as the extension:

``` text
TS-JEPA baseline
+
Gilbert-Elliott burst channel
+
burst masking
+
Burst Position Encoding
+
burst-aware JEPA objective
```

The baseline must be treated as frozen once validated so that the
GE-JEPA contribution can be measured cleanly.

------------------------------------------------------------------------

# 30. Source Status

This document is based primarily on:

-   The supplied TS-JEPA paper PDF.
-   The current reproduction/implementation notes.
-   Corrections identified during the architecture and implementation
    cross-check.

Where the source material does not fully specify a value or mechanism,
this document intentionally marks it as unresolved instead of
fabricating a paper-exact answer.
