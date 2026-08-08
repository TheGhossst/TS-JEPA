# TS-JEPA Paper-Faithful Architecture
## Exact reproduction specification based on the published IEEE Internet of Things Journal version

**Base paper:** Abanoub M. Girgis, Alvaro Valcarce, and Mehdi Bennis,  
**“Time-Series JEPA for Predictive Remote Control Under Capacity-Limited Networks,”**  
IEEE Internet of Things Journal, Vol. 13, No. 7, 1 April 2026, pp. 14617–14632.  
DOI: 10.1109/JIOT.2025.3650435

> **Scope of this document**
>
> This document is a paper-faithful reproduction specification for the **published 2026 IEEE version** of TS-JEPA.
>
> It deliberately distinguishes:
>
> - **PAPER-SPECIFIED** — explicitly stated in the published paper.
> - **PAPER-IMPLIED** — follows directly from the equations/figures, but a low-level implementation detail is not explicitly stated.
> - **IMPLEMENTATION CHOICE** — necessary for coding where the paper leaves a detail unspecified.
>
> Do not silently replace a paper specification with a more convenient implementation.

---

# 1. Paper-Faithfulness Rules

## 1.1 Source version

This specification follows the **published 2026 IEEE version**, not the earlier 2024 arXiv/preprint version.

This distinction matters because the published paper changed several details of the TS-JEPA architecture, training configuration, data generation, and system formulation.

## 1.2 What must not be invented

The following are **not** to be introduced as paper facts unless separately verified from the paper:

- a linearized CartPole state-transition matrix `A`
- an input matrix `B`
- an LQR controller
- a particular `Q` or `R` matrix
- a particular CartPole physics implementation
- a 512-unit TS-JEPA predictor
- a 512-unit encoder block
- a 512-D embedding
- an MAE/L1 semantic-actor objective
- a desired-state vector concatenated to the semantic actor input
- a 6-channel frame tensor as the explicitly specified paper input
- a 25-trajectory test set
- 20,000 / 4,000 / 2,500 samples as the published-paper dataset specification

Where the paper is silent, this document says so explicitly.

---

# 2. Overall Published TS-JEPA Framework

The published framework contains three main learned components:

1. **Context encoder**
2. **Target encoder**
3. **Predictor**

After TS-JEPA training, a separate:

4. **Semantic actor**

maps semantic embeddings to control commands.

The broader framework also contains a **channel-aware scheduler**, but that scheduler is a networking component rather than part of the TS-JEPA neural architecture.

High-level flow:

```text
                    HIGH-DIMENSIONAL DEVICE STATE
                               x_i,k
                                 |
                                 v
                        +----------------+
                        | Context Encoder|
                        |     Ψ_θ        |
                        +----------------+
                                 |
                                 v
                         Semantic embedding
                              z_i,k
                                 |
                                 v
                     +----------------------+
                     |      Predictor       |
                     |       P_ϕ            |
                     | conditioned on       |
                     | predicted commands   |
                     +----------------------+
                                 |
                                 v
                    Predicted future embeddings
                  z~_i,k+1 ... z~_i,k+Kp
                                 |
                                 | cosine similarity
                                 v
                    Target future embeddings
                  z_i,k+1 ... z_i,k+Kp
                                 ^
                                 |
                         +---------------+
                         | Target Encoder|
                         |     Ψ_θbar     |
                         +---------------+
                                 ^
                                 |
                    Future device states
                  x_i,k+1 ... x_i,k+Kp
```

**PAPER-SPECIFIED:** The context encoder maps the current high-dimensional state to a low-dimensional embedding; the target encoder processes future states; the predictor predicts future embeddings conditioned on the current embedding and predicted control commands. [IEEE paper, Sec. III-A, Eq. (12)–(14)]

---

# 3. System / CartPole Simulation

## 3.1 Device state

The paper denotes the device state as:

```text
x_i,k
```

where:

- `i` = device index
- `k` = time index

The general system model uses nonlinear dynamics:

```text
x_i,k+1 = f_i(x_i,k, u_i,k) + n_s,k
```

where:

- `f_i` = nonlinear dynamics
- `u_i,k` = control command
- `n_s,k` = process noise

**PAPER-SPECIFIED:** The published paper does not replace the system with a linear `A/B` model.

---

## 3.2 Visual state representation

**PAPER-SPECIFIED:**

Each inverted cart-pole state is represented by an **RGB frame**.

The simulation uses:

```text
sampling interval τ_o = 1 ms
number of time steps = 100
```

Therefore, the stated trajectory duration implied by these two quantities is:

```text
100 × 1 ms = 100 ms = 0.1 s
```

The paper does not describe this as a 1-second or 100-second trajectory.

---

## 3.3 Control command

The control command applies a **horizontal force to the cart**.

**PAPER-SPECIFIED:**

```text
u_max = +20 N
u_min = -20 N
```

The commands are generated using a **nonlinear control policy**.

Do not substitute an LQR controller when claiming a strict reproduction.

---

## 3.4 Desired state

The control-performance metric uses:

```text
x_d = desired cart position
θ_i,k = pendulum angle
```

The scoring condition is:

```text
|x_i,k - x_d| <= 0.05
AND
|θ_i,k| <= 0.05
```

with score:

```text
R_i,k = 1  if both conditions are satisfied
        0  otherwise
```

**PAPER-SPECIFIED:** The desired position is used in the control-performance evaluation.

**IMPORTANT:** The paper does **not** state that the desired state is concatenated to the semantic actor's neural-network input.

---

# 4. Temporal Configuration

## 4.1 Sampling

```yaml
sampling_interval: 1 ms
trajectory_steps: 100
```

**PAPER-SPECIFIED**

---

## 4.2 Consecutive-frame parameter κ

The paper defines:

```text
κ = number of consecutive frames
```

and explicitly evaluates:

```text
κ = 2
```

for the main TS-JEPA comparison.

It also compares supervised-learning baselines with:

```text
κ = 2
κ = 4
```

and a generative autoencoder with:

```text
κ = 2
```

**PAPER-SPECIFIED**

### Important implementation boundary

The formal TS-JEPA architecture describes the context encoder as operating on:

```text
x_i,k
```

while the experiments explicitly discuss performance using `κ` consecutive frames.

The published paper does **not explicitly state** the exact tensor operation used to combine the `κ=2` RGB frames.

Therefore:

```text
κ = 2
```

is paper-specified, but:

```text
[B, 6, H, W]
```

obtained by channel-concatenating two RGB frames is an **IMPLEMENTATION CHOICE**, not a paper-stated tensor specification.

Do not describe 6-channel concatenation as explicitly specified by the paper.

---

# 5. Input Preprocessing

The paper specifies the following preprocessing for RGB frames.

## 5.1 Color jittering

Random adjustments:

```text
brightness = 0.05
contrast   = 0.10
saturation = 0.10
hue        = 0.05
```

The adjustments are applied in random order for each patch.

**PAPER-SPECIFIED**

---

## 5.2 Color dropping

Probability:

```text
p = 0.05
```

The RGB frame is converted to grayscale using the luma component.

**PAPER-SPECIFIED**

---

## 5.3 Normalization

Per-channel normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**PAPER-SPECIFIED**

---

## 5.4 Resizing

Frames are resized to:

```text
64 × 128
```

using:

```text
5 × 5 Gaussian kernel
σ randomly sampled from [0.1, 0.2]
```

**PAPER-SPECIFIED**

---

## 5.5 Testing preprocessing

During TS-JEPA testing:

```text
resize
+
normalization
```

are applied consistently with training.

The paper does not state that stochastic augmentation must remain active during testing.

---

# 6. TS-JEPA Architecture

## 6.1 Context / online encoder

The context encoder is:

```text
Ψ_θ(·)
```

and maps:

```text
x_i,k
    ↓
Ψ_θ
    ↓
z_i,k
```

**PAPER-SPECIFIED**

The published paper describes it as a **deep convolutional residual network (ResNet)**.

The stated layer widths are:

```text
64
128
256
```

with:

```text
Batch Normalization
+
ReLU
```

associated with the layers.

**PAPER-SPECIFIED**

### Important boundary

The paper does not provide enough low-level detail in the architecture text to uniquely reconstruct:

- exact initial convolution kernel size
- exact number of residual blocks
- exact stride of every block
- exact padding configuration
- exact placement of global average pooling
- exact intermediate tensor shapes

Therefore those details must not be falsely labeled as paper-specified.

---

## 6.2 Embedding dimension

The paper describes `z_i,k` as a low-dimensional semantic embedding, but it **does not explicitly state the embedding dimensionality or define a dedicated symbol for it**.

The published architecture provides two separate facts:

1. the stated ResNet layer widths end at **256**; and
2. the predictor has an output layer of **256 neurons**, while its output is the predicted future embedding.

Together, these facts strongly imply:

```text
z_i,k ≈ 256-dimensional
```

**PAPER-IMPLIED** — not explicitly stated as an embedding dimension in the paper.

The paper also evaluates different embedding dimensions experimentally and reports that larger dimensions can improve prediction accuracy while increasing communication cost.

---

# 7. Target Encoder

The target encoder is:

```text
Ψ_θbar(·)
```

It mirrors the context encoder architecture.

Initialization:

```text
θ_bar ← θ
```

**PAPER-SPECIFIED**

The target encoder processes future states:

```text
x_i,k+1
x_i,k+2
...
x_i,k+Kp
```

and produces:

```text
z_i,k+1
z_i,k+2
...
z_i,k+Kp
```

**PAPER-SPECIFIED**

---

## 7.1 Stop-gradient

The target branch does not receive gradients.

The paper explicitly states that gradients are blocked through the target branch.

**PAPER-SPECIFIED**

---

## 7.2 EMA update

After updating the context encoder and predictor:

```text
θ_bar ← η θ_bar + (1 - η) θ
```

The published Table II specifies:

```text
η = 0.99
```

**PAPER-SPECIFIED**

---

# 8. Predictor

The predictor is:

```text
P_ϕ(·)
```

Its purpose is to model nonlinear embedding evolution.

It receives:

```text
current embedding
+
predicted control commands
```

and produces future semantic embeddings.

Formal paper definition:

```text
(z~_i,k+1, ..., z~_i,k+Kp)
 =
P_ϕ(
    z_i,k,
    u~_i,k, ...,
    u~_i,k+Kp-1
)
```

**PAPER-SPECIFIED**

---

## 8.1 Predictor architecture

The paper specifies:

```text
MLP
hidden layer: 1024
output layer: 256
```

with ReLU activation described for the hidden layer.

Therefore the structural form is:

```text
[embedding + control-command input]
             |
             v
        Linear(1024)
             |
           ReLU
             |
             v
        Linear(256)
             |
             v
      predicted embedding
```

**PAPER-SPECIFIED**

### Input dimension

The paper's nomenclature defines:

```text
q = control-command dimension
```

The paper does **not** define a symbol `d` for embedding dimension.

The predictor therefore conceptually receives the current embedding together with predicted control-command information. The embedding dimensionality is strongly inferred to be 256 from the predictor's 256-neuron output, but this is **PAPER-IMPLIED**, not a nomenclature definition.

for a single current embedding/command step.

For the CartPole horizontal-force case, the control command is a scalar force, but the published paper does not provide a low-level neural-network input-shape specification.

Do not hard-code:

```text
257
```

as a universal paper-defined input dimension.

---

# 9. Autoregressive Prediction

The predictor operates autoregressively.

Conceptually:

```text
z_k
 |
 +---- predicted command u~_k
 |
 v
P_ϕ
 |
 v
z~_(k+1)
 |
 +---- predicted command u~_(k+1)
 |
 v
P_ϕ
 |
 v
z~_(k+2)
 |
 ...
```

The paper explicitly states that future embeddings are predicted using predicted control commands and that prediction errors accumulate slightly over longer horizons due to autoregressive prediction.

**PAPER-SPECIFIED**

---

# 10. Prediction Horizon

The published paper evaluates TS-JEPA over a:

```text
15-step prediction horizon
```

The abstract explicitly reports robustness on 15-step horizons.

The paper uses:

```text
Kp
```

as the prediction-horizon variable.

### Fidelity label

```text
15-step horizon:
PAPER-SPECIFIED FOR THE REPORTED EVALUATION

Kp = 15 as an unconditional architectural constant:
DO NOT CLAIM THIS UNLESS YOUR IMPLEMENTATION IS SPECIFICALLY
REPRODUCING THE 15-STEP EXPERIMENT.
```

---

# 11. JEPA Loss

The paper uses cosine similarity between predicted and target embeddings.

Conceptually:

```text
predicted embedding
        vs.
target embedding
        ↓
cosine similarity
```

The target branch is stop-gradient.

The optimization updates:

```text
θ
ϕ
```

while the target parameters are updated by EMA.

**PAPER-SPECIFIED**

For implementation, the loss can be represented as the negative cosine similarity or an equivalent minimizing form, provided the optimization direction is preserved.

The paper itself formulates the objective using cosine similarity; do not claim that the exact Python expression `1 - cosine_similarity()` is literally printed in the paper.

---

# 12. TS-JEPA Training Algorithm

The paper's Algorithm 1 can be implemented conceptually as:

```text
Initialize:
    context encoder Ψ_θ
    predictor P_ϕ

Set:
    target encoder Ψ_θbar ← Ψ_θ

For each training step k:

    1. Encode current state:
         z_k = Ψ_θ(x_k)

    2. Encode future states with target encoder:
         z_(k+j) = Ψ_θbar(x_(k+j))
         for j = 1 ... Kp

    3. Predict future embeddings:
         z~_(k+1 ... k+Kp)
         = P_ϕ(z_k, predicted commands)

    4. Compute cosine-similarity objective

    5. Gradient-update:
         θ
         ϕ

    6. EMA-update:
         θbar ← η θbar + (1-η) θ
```

**PAPER-SPECIFIED**

---

# 13. TS-JEPA Training Hyperparameters

From the published Table II:

```yaml
ts_jepa:
  learning_rate: 0.2
  batch_size: 256
  epochs: 150
  optimizer: SGD
  weight_decay: 0.0004
  ema_decay: 0.99
  lr_decay_factor: 0.99
  lr_decay_interval_epochs: 20
```

**PAPER-SPECIFIED**

The paper states that the learning rate decays by a factor of `0.99` every `20` epochs.

---

# 14. TS-JEPA Dataset

The paper uses a separate dataset for TS-JEPA:

```text
D_s
```

It contains RGB frames paired with their corresponding control commands.

**PAPER-SPECIFIED:**

```text
training trajectories = 200
testing trajectories  = 40
```

There is no published-paper requirement here for a separate 25-trajectory test set.

---

# 15. Validation / Early Stopping

The paper states:

```text
early stopping based on validation performance
```

However, the published trajectory counts explicitly given for the TS-JEPA dataset are:

```text
200 training trajectories
40 testing trajectories
```

The paper does not give a separate trajectory count for the validation set in the same dataset description.

Therefore:

```text
validation mechanism = PAPER-SPECIFIED
exact validation trajectory count = NOT SPECIFIED
```

Do not invent a 40/25/etc. split.

---

# 16. Training Repetitions

The paper states:

```text
each experiment is repeated 5 times
best results are reported
```

**PAPER-SPECIFIED**

This is an experimental protocol, not part of the neural-network architecture.

---

# 17. Semantic Actor

After TS-JEPA is trained, a separate semantic actor is trained.

The actor is:

```text
C_ε
```

Its purpose is to map the semantic embedding directly to the control command.

Flow:

```text
z_i,k
  |
  v
Semantic Actor C_ε
  |
  v
u~_i,k
```

The actor can receive either:

```text
embedding from the context encoder
```

or:

```text
embedding predicted by the TS-JEPA predictor
```

**PAPER-SPECIFIED**

---

# 18. Semantic Actor Dataset

The actor uses a separate dataset:

```text
D_a = {(z_i,k, u_i,k)}
```

**PAPER-SPECIFIED:**

```text
training trajectories = 100
testing trajectories  = 20
```

The dataset contains:

```text
low-dimensional embedding
+
corresponding control command
```

---

# 19. Semantic Actor Architecture

The published paper specifies an MLP with two hidden layers:

```text
hidden layer 1 = 1024
hidden layer 2 = 256
```

Each hidden layer uses ReLU.

Therefore:

```text
≈256-D semantic embedding (PAPER-IMPLIED)
          |
          v
      Linear(1024)
          |
        ReLU
          |
          v
      Linear(256)
          |
        ReLU
          |
          v
      Control output
```

**PAPER-SPECIFIED**

The paper does not state that a desired-state vector is concatenated with the embedding.

Therefore:

```text
[z, desired_state]
```

must NOT be described as the paper architecture.

---

# 20. Semantic Actor Output

The output is the predicted control command:

```text
u~_i,k
```

For the simulated inverted cart-pole, this represents the horizontal force command.

The paper's general notation allows `q` control dimensions.

Do not hard-code a multi-dimensional action vector unless your simulation configuration explicitly requires it.

---

# 21. Semantic Actor Loss

The actor minimizes the average squared L2 distance between predicted and target control commands:

```text
L_actor
=
(1 / K_a)
Σ ||u_i,k - u~_i,k||²_2
```

The training section explicitly describes this as **MSE**.

**PAPER-SPECIFIED**

Therefore:

```python
loss = MSE(predicted_command, target_command)
```

is the appropriate implementation.

Do NOT use MAE/L1 if claiming strict reproduction.

---

# 22. Semantic Actor Hyperparameters

From published Table III:

```yaml
semantic_actor:
  learning_rate: 0.006
  batch_size: 200
  epochs: 300
  optimizer: AdamW
  dropout: 0.2
```

**PAPER-SPECIFIED**

The paper also states that the two hidden layers use ReLU.

---

# 23. Control-Command Normalization

The paper states that control commands are preprocessed using:

```text
z-score normalization
```

for stable TS-JEPA training.

Implementation form:

```text
u_norm = (u - μ) / σ
```

where the statistics should correspond to the training data.

**PAPER-SPECIFIED:** z-score normalization.

**IMPLEMENTATION CHOICE:** Exact code structure for storing/loading `μ` and `σ`.

---

# 24. Inference Architecture

After training:

```text
DEVICE / SENSOR
---------------

RGB state x_k
      |
      v
Context Encoder Ψ_θ
      |
      v
z_k
      |
      | transmit semantic embedding
      v

REMOTE CONTROLLER
-----------------

Received z_k
      |
      +----------------------+
      |                      |
      v                      v
Semantic Actor          Predictor P_ϕ
      |                      |
      v                      v
u~_k                 z~_(k+1), ...
```

When a fresh semantic embedding is received, the semantic actor can directly compute the control command.

When fresh communication is unavailable, the predictor can generate future embeddings, which can then be supplied to the semantic actor.

**PAPER-SPECIFIED**

---

# 25. Communication / Scheduling Layer

The published framework adds a channel-aware scheduler around TS-JEPA.

The scheduler considers:

```text
channel conditions
+
Age of Information (AoI)
+
available resource blocks
+
transmission reliability
+
transmission power
```

The scheduler can select up to:

```text
J devices
```

per time step according to the paper's optimization formulation.

This component is **not part of the TS-JEPA neural architecture itself**.

For a first TS-JEPA reproduction, implement it separately.

---

# 26. Channel-Aware Scheduler — High-Level Flow

```text
For each time step k:

    observe:
        channel H_i,k
        AoI β_i,k
        virtual queue Q_i,k

    calculate:
        required transmission power p_req_i,k

    if p_req_i,k > p_max:
        mark device infeasible

    otherwise:
        calculate drift-plus-penalty index S_i,k

    select:
        up to J devices with largest positive S_i,k

    scheduled device:
        α_i,k = 1
        p_i,k = p_req_i,k
        AoI resets

    unscheduled device:
        α_i,k = 0
        p_i,k = 0
        AoI increments

    update virtual queue
```

**PAPER-SPECIFIED**

This scheduler should be treated as a separate research layer from the TS-JEPA neural reproduction.

---

# 27. Wireless Simulation Parameters

The published Table IV specifies:

```yaml
wireless:
  hall_size: "300 × 150 m²"
  room_height: "6 m"
  bs_height: "10.0 m"
  device_height: "1.5 m"
  carrier_frequency: "3.75 GHz"
  total_bandwidth: "20 MHz"
  clutter_height: "3 m"
  clutter_size: "2.0 m"
  clutter_density: "60%"
  2d_distance: "50 m"
  noise_power: "-95 dB"
```

The target SNR values evaluated are:

```text
γ_th ∈ {5, 10, 20} dB
```

**PAPER-SPECIFIED**

These parameters belong to the wireless-network evaluation, not to the core TS-JEPA neural model.

---

# 28. Evaluation Metrics

The paper uses five major evaluation categories.

## 28.1 Encoder performance

t-SNE visualization of learned embeddings.

Purpose:

```text
evaluate whether semantically similar states form meaningful clusters
```

---

## 28.2 Temporal / spatial consistency

MAPE between consecutive frames.

---

## 28.3 Prediction accuracy

Normalized mean absolute error:

```text
NMAE
```

between predicted and ground-truth control commands over the prediction horizon.

---

## 28.4 Control performance

Binary control score:

```text
R_i,k = 1
if:
    |x_i,k - x_d| <= 0.05
    AND
    |θ_i,k| <= 0.05

otherwise:
    R_i,k = 0
```

---

## 28.5 Communication efficiency

Number of communication bits required to transmit:

```text
high-dimensional state
```

or:

```text
semantic embedding
```

per step.

---

# 29. Main Baselines

The published paper evaluates against:

### Control baselines

1. Optimal nonlinear control policy
2. Supervised learning model
3. Generative autoencoder model

### Scheduling baselines

4. Round-robin scheduling
5. Opportunistic scheduling

These baselines are part of the evaluation framework, not required for the minimum TS-JEPA training implementation.

---

# 30. Minimum Reproduction Scope

If the immediate goal is:

> “Reproduce TS-JEPA before adding GE-JEPA modifications.”

then implement in this order.

## Phase 1 — CartPole data generation

```text
Inverted CartPole
    |
    +-- RGB frames
    |
    +-- nonlinear control policy
    |
    +-- force range [-20, +20] N
    |
    +-- τ0 = 1 ms
    |
    +-- 100 time steps
```

---

## Phase 2 — preprocessing

```text
RGB
 |
 +-- color jitter
 |
 +-- color dropping p=0.05
 |
 +-- Gaussian resize → 64×128
 |
 +-- ImageNet-style normalization
 |
 +-- command z-score normalization
```

---

## Phase 3 — context encoder

```text
RGB state
    |
    v
ResNet
    |
    +-- 64
    +-- 128
    +-- 256
    |
    v
≈256-D embedding
(PAPER-IMPLIED)
```

---

## Phase 4 — target encoder

```text
same architecture as context encoder
        |
        v
future embeddings

NO gradient
EMA from context encoder
η = 0.99
```

---

## Phase 5 — predictor

```text
current embedding
        +
predicted control command
        |
        v
MLP
    1024
      |
    ReLU
      |
     256
      |
      v
predicted future embedding
```

---

## Phase 6 — JEPA training

```text
context embedding
        |
        v
predict future embedding
        |
        | cosine similarity
        v
target future embedding
```

Training:

```yaml
optimizer: SGD
learning_rate: 0.2
batch_size: 256
epochs: 150
weight_decay: 0.0004
ema_decay: 0.99
lr_decay_factor: 0.99
lr_decay_interval: 20 epochs
```

---

## Phase 7 — semantic actor

```text
≈256-D embedding (PAPER-IMPLIED)
      |
      v
    1024
      |
    ReLU
      |
     256
      |
    ReLU
      |
      v
control command
```

Training:

```yaml
optimizer: AdamW
learning_rate: 0.006
batch_size: 200
epochs: 300
dropout: 0.2
loss: MSE
```

---

# 31. Paper-Faithful Configuration Summary

```yaml
paper:
  version: "IEEE Internet of Things Journal 2026"

simulation:
  environment: "inverted cart-pole"
  state_representation: "RGB frame"
  sampling_interval_ms: 1
  time_steps: 100
  control_policy: "nonlinear"
  control_min_N: -20
  control_max_N: 20

input:
  frame_channels: 3
  resize: [64, 128]
  color_jitter:
    brightness: 0.05
    contrast: 0.10
    saturation: 0.10
    hue: 0.05
  color_drop_probability: 0.05
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  gaussian_kernel: [5, 5]
  gaussian_sigma_range: [0.1, 0.2]

ts_jepa:
  # Embedding dimension is inferred from the 256-neuron predictor output.
  # PAPER-IMPLIED, not explicitly stated by the paper.
  embedding_dim: 256
  context_encoder:
    architecture: "ResNet"
    layer_widths: [64, 128, 256]
    activation: "ReLU"
    batch_normalization: true

  target_encoder:
    architecture: "same as context encoder"
    stop_gradient: true
    ema_decay: 0.99

  predictor:
    architecture: "MLP"
    hidden_dim: 1024
    output_dim: 256
    hidden_activation: "ReLU"

  prediction:
    horizon_variable: "Kp"
    reported_evaluation_horizon: 15
    autoregressive: true
    conditioning: "current embedding + predicted control commands"

  training:
    optimizer: "SGD"
    learning_rate: 0.2
    batch_size: 256
    epochs: 150
    weight_decay: 0.0004
    lr_decay_factor: 0.99
    lr_decay_every_epochs: 20
    loss: "cosine similarity"
    repetitions: 5

  dataset:
    training_trajectories: 200
    testing_trajectories: 40

semantic_actor:
  input: "semantic embedding"
  architecture: "MLP"
  hidden_dims: [1024, 256]
  activation: "ReLU"
  loss: "MSE"
  optimizer: "AdamW"
  learning_rate: 0.006
  batch_size: 200
  epochs: 300
  dropout: 0.2

  dataset:
    training_trajectories: 100
    testing_trajectories: 20

evaluation:
  main_frame_setting_kappa: 2
  prediction_horizon_steps: 15
  target_snr_db: [5, 10, 20]
```

---

# 32. Explicitly Unspecified by the Published Paper

The following details should remain marked as **implementation decisions** unless additional author/code material is found:

```text
1. Exact CartPole physics parameters:
   cart mass, pole mass, pole length, gravity, etc.

2. Exact numerical nonlinear-control-policy implementation.

3. Exact neural-network ResNet block count and convolution
   kernel/stride/padding configuration.

4. Exact tensor construction for κ=2 consecutive RGB frames.

5. Exact predictor input tensor layout for multiple prediction steps.

6. Exact handling of batch windows near trajectory boundaries.

7. Exact validation-trajectory count.

8. Exact random augmentation implementation details beyond
   the parameters stated in the paper.

9. Exact dropout placement in the semantic actor.

10. Exact action-output activation, if any.

11. Exact random seeds.

12. Exact software framework/version.

13. Exact optimizer epsilon/beta settings.

14. Exact checkpoint-selection implementation.

15. Exact implementation of the nonlinear control policy.
```

These should not be invented and then labeled “paper-faithful.”

---

# 33. Critical Reproduction Warnings

## DO NOT use

```text
1 Hz
```

as the paper's sampling rate.

Use:

```text
1 ms
```

---

## DO NOT use

```text
±25 N
```

Use:

```text
±20 N
```

---

## DO NOT replace the nonlinear control policy with LQR

The published simulation explicitly uses a nonlinear control policy.

---

## DO NOT use the older 512-unit predictor

Published 2026 paper:

```text
predictor hidden = 1024
output = 256
```

---

## DO NOT use

```text
weight_decay = 0.004
```

Published Table II:

```text
weight_decay = 0.0004
```

---

## DO NOT use

```text
LR decay = 0.9 every 10 epochs
```

Published paper:

```text
LR decay = 0.99 every 20 epochs
```

---

## DO NOT use MAE for the semantic actor

Published paper:

```text
MSE
```

---

## DO NOT use Adam for the semantic actor

Published Table III:

```text
AdamW
```

---

## DO NOT add desired-state variables to the actor input

The paper defines the actor dataset as:

```text
(z_i,k, u_i,k)
```

and describes the actor as mapping semantic embeddings to control commands.

---

# 34. Strict Paper-Faithful Architecture Diagram

```text
                         INVERTED CART-POLE
                                |
                                | RGB state x_k
                                v
                     +-----------------------+
                     |     Context Encoder    |
                     |        Ψ_θ             |
                     |                       |
                     |       ResNet          |
                     |   64 → 128 → 256     |
                     |       + BN + ReLU     |
                     +-----------------------+
                                |
                                v
                         z_k ∈ R^256
                                |
                    +-----------+-----------+
                    |                       |
                    |                       |
                    v                       v
             Semantic Actor            Predictor P_ϕ
                    |                       |
                    |                 z_k + predicted
                    |                 control commands
                    |                       |
                    |                       v
                    |                MLP: 1024 → 256
                    |                       |
                    |                       v
                    |                 z~_(k+1...k+Kp)
                    |                       |
                    |                       |
                    |                cosine similarity
                    |                       |
                    |                       v
                    |               Target embeddings
                    |                       ^
                    |                       |
                    |                Target Encoder
                    |                   Ψ_θbar
                    |                       |
                    |                EMA, η = 0.99
                    |                       |
                    |                       ^
                    |                       |
                    |          x_(k+1)...x_(k+Kp)
                    |
                    v
             control command u~_k
```

---

# 35. Reproduction Boundary for GE-JEPA

This document is the **baseline TS-JEPA only**.

Do not add:

```text
Gilbert-Elliott masking
Burst Position Encoding
burst-aware loss
packet-loss-conditioned predictor
```

to this baseline.

The correct research sequence is:

```text
                    TS-JEPA PAPER
                         |
                         v
             faithful reproduction
                         |
                         v
             reproduce baseline results
                         |
                         v
             verify encoder/predictor/
             semantic-actor behavior
                         |
                         v
                    GE-JEPA
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      GE masking       BPE        burst-aware loss
```

This preserves a clean experimental baseline for the later GE-JEPA contribution.

---

# 36. Final Paper-Faithfulness Checklist

Before calling the implementation “TS-JEPA reproduced,” verify:

### System

- [ ] Inverted cart-pole
- [ ] RGB state representation
- [ ] 1 ms sampling interval
- [ ] 100 time steps
- [ ] nonlinear control policy
- [ ] control range [-20, +20] N

### Input

- [ ] color jitter parameters match
- [ ] color dropping probability = 0.05
- [ ] normalization mean/std match
- [ ] resize = 64 × 128
- [ ] Gaussian kernel = 5 × 5
- [ ] sigma sampled from [0.1, 0.2]

### TS-JEPA

- [ ] context encoder = ResNet
- [ ] encoder widths = 64, 128, 256
- [ ] BN + ReLU
- [ ] embedding dimension inferred as 256 (PAPER-IMPLIED, not explicitly stated)
- [ ] target encoder mirrors context encoder
- [ ] target branch stop-gradient
- [ ] EMA = 0.99
- [ ] predictor hidden = 1024
- [ ] predictor output = 256
- [ ] autoregressive prediction
- [ ] conditioning on predicted control commands
- [ ] cosine-similarity objective
- [ ] SGD
- [ ] LR = 0.2
- [ ] batch = 256
- [ ] epochs = 150
- [ ] weight decay = 0.0004
- [ ] LR decay = 0.99 every 20 epochs

### Dataset

- [ ] TS-JEPA = 200 train trajectories
- [ ] TS-JEPA = 40 test trajectories
- [ ] separate actor dataset
- [ ] actor = 100 train trajectories
- [ ] actor = 20 test trajectories

### Semantic Actor

- [ ] embedding input
- [ ] hidden = 1024
- [ ] hidden = 256
- [ ] ReLU
- [ ] MSE
- [ ] AdamW
- [ ] LR = 0.006
- [ ] batch = 200
- [ ] epochs = 300
- [ ] dropout = 0.2

### Evaluation

- [ ] κ = 2 main TS-JEPA comparison
- [ ] 15-step prediction evaluation
- [ ] t-SNE encoder evaluation
- [ ] NMAE control prediction metric
- [ ] control score
- [ ] communication-bit measurement

---

# 37. Source Mapping

Primary source used for this specification:

**Girgis, Valcarce, Bennis — “Time-Series JEPA for Predictive Remote Control Under Capacity-Limited Networks,” IEEE Internet of Things Journal, 2026.**

Relevant published-paper locations:

- System and TS-JEPA formulation: pp. 14622–14623
- TS-JEPA / semantic actor architecture: pp. 14623
- Channel-aware scheduling: pp. 14623–14625
- Simulation/data generation/training: pp. 14625–14626
- Preprocessing/evaluation: p. 14626
- Evaluation and baselines: pp. 14627–14630
- Published TS-JEPA hyperparameters: Table II, p. 14625
- Published semantic-actor hyperparameters: Table III, p. 14625
- Wireless parameters: Table IV, p. 14626

---

## End of paper-faithful specification
