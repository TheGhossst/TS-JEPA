# Time-Series Joint-Embedding Predictive Architecture (TS-JEPA) Block-by-Block Architecture Map (v2)

This document provides a highly rigorous, block-by-block technical mapping of the **TS-JEPA** framework from the paper *"Time-Series JEPA for Predictive Remote Control Under Capacity-Limited Networks"*. It outlines every stage of the pipeline—from raw physical data generation through neural encoding, prediction, and actor inference—evaluating shapes, parameter states, exact architectures, and necessary implementation assumptions.

*This version (v2) incorporates critical implementation notes and debugging warnings to prevent silent failure modes during reproduction.*

---

```
  [1. Input: Raw State & Action]
                 ↓
      [2. Preprocessing & Aug]
                 ↓
      [3. Context Encoder] ──(EMA)──> [4. Target Encoder]
                 ↓                            ↓
          [z_k (Embedding)]            [z_{k+1...K_p} (Targets)]
                 ↓                            │
       [5. Masking (Temporal)]                │
                 ↓                            │
  [6. Predictor] <── [u_k (Actions)]          │
                 ↓                            │
    [~z_{k+1...K_p} (Predictions)]            │
                 │                            │
                 └─────> [7. Latent Loss] <───┘
                            (Cosine)
                 ┌────────────────────────────┐
                 ▼                            ▼\
      [8. Semantic Actor]             [9. Offline Training]
                 ↓                            ↓
         [u_k (Force Command)]        [10. Run-time Inference]
```

---

## 1. Input (Raw State & Action)

### **Role in the System**
This is the data generation stage, representing the physical state of the unstable dynamical system (the cart-pole) and the horizontal force commands applied by the actuators [21, 56].

*   **Input shape?**
    *   **Visual State ($x_{i,k}$)**: The raw visual state. Before preprocessing, this is a high-resolution color (RGB) camera frame. The original resolution before resizing is not explicitly specified in the paper, but we can assume a standard sensor resolution like $480 \times 640 \times 3$ or $1080 \\times 1920 \\times 3$.
    *   **Control Command ($u_{i,k}$)**: $(q,) = (1,)$ for the cart-pole system, representing a single continuous scalar horizontal force [56].
*   **Output shape?** Same as the input.
*   **Trainable?** No. This represents raw physical telemetry and mathematical control commands.
*   **Frozen?** Yes. It represents the physical reality of the environment.
*   **Paper gives exact architecture?** No. The paper treats this as the physical environment block. It specifies that the state is represented by an RGB frame captured by a sensor at a fixed sampling interval of $\tau_o = 1\text{ ms}$ [56].
*   **Need assumptions?** Yes. We must assume the camera captures 3 color channels (RGB) and that the raw frame rate matches the $1\text{ ms}$ sampling clock of the control loop [56].

---

## 2. Preprocessing & Augmentation

### **Role in the System**
This stage normalizes continuous control forces and processes raw visual frames to maximize spatial and temporal diversity, which prevents representation collapse during self-supervised learning [60, 61, 72].

*   **Input shape?**
    *   **Visual Frame**: Raw RGB image of arbitrary resolution.
    *   **Control Command ($u_{i,k}$)**: Raw continuous force value in the range $[-20\text{ N}, 20\text{ N}]$ [56].
*   **Output shape?**
    *   **Visual Frame**: $(3, 64, 128)$ assuming channel-first layout (PyTorch standard) [61].
    *   **Control Command**: A normalized scalar of shape $(1,)$ [61].
*   **Trainable?** No. It consists of deterministic mathematical transformations and stochastic augmentations [60, 61].
*   **Frozen?** Yes. The preprocessing operations are non-parametric.
*   **Paper gives exact architecture?** Yes. The paper explicitly documents the preprocessing and image augmentation steps in Section IV-A [60, 61]:
    1.  **Color Jittering (Training Only)**: Randomly adjusts brightness (0.05), contrast (0.1), saturation (0.1), and hue (0.05) [60].
    2.  **Color Dropping (Training Only)**: Converts frames to grayscale with a 0.05 probability [60].
    3.  **Normalization**: Channel-wise subtraction of mean $[0.485, 0.456, 0.406]$ and division by standard deviation $[0.229, 0.224, 0.225]$ [61].
    4.  **Resizing**: Downsamples frames to $64 \times 128$ using a $5 \times 5$ Gaussian kernel with standard deviation randomly sampled from $[0.1, 0.2]$ [61].
    5.  **Control Commands**: Z-score normalization [61].
*   **Need assumptions?** Yes. We assume PyTorch-style channel ordering $(C, H, W)$ for downstream convolutional layers. During the testing and deployment phases, we assume that color jittering and color dropping are disabled, applying only Gaussian resizing and normalization [61].

---

## 3. Context Encoder ($\Psi_\theta$)

### **Role in the System**
The local neural network running on the device. It compresses the high-dimensional visual frame into a compact, low-dimensional semantic embedding, discarding task-irrelevant background visual noise [14, 35, 74].

*   **Input shape?** $(3, 64, 128)$ [61].
*   **Output shape?** $(256,)$, representing the low-dimensional semantic embedding vector $z_{i,k}$ [35, 59].
*   **Trainable?** Yes. It is parameterized by a set $\theta$ of learnable weights [35].
*   **Frozen?** No. It is actively updated using gradient descent during TS-JEPA training [37]. (It is frozen during the Semantic Actor training and run-time inference phases) [41, 44].
*   **Paper gives exact architecture?** Partial details. The paper states: *\"The TS-JEPA encoder adopts a deep convolutional residual network (ResNet) architecture with layers of 64, 128, and 256 neurons, each followed by batch normalization and rectifier linear unit (ReLU) activation\"* [59].
*   **Need assumptions?** Yes. While the paper defines the channel sizes of the ResNet layers (64, 128, 256), we must assume:\n    *   The exact block structures (e.g., number of convolutional layers per block, kernel size $3 \times 3$, stride, padding).\n    *   The presence of a Global Average Pooling (GAP) layer or a flattening layer at the output to convert the final convolutional feature maps into a flat vector of shape $(256,)$.

---

## 4. Target Encoder ($\Psi_{\bar{\theta}}$)

### **Role in the System**
A mirror model of the Context Encoder that encodes a sequence of future unmasked frames to provide stable target embeddings for the predictor, preventing representation collapse [38].

*   **Input shape?** A sequence of future preprocessed frames $(x_{i,k+1}, \dots, x_{i,k+K_p})$, shape $(K_p, 3, 64, 128)$ where $K_p = 15$ is the prediction horizon [35].
*   **Output shape?** A sequence of future target embeddings $(z_{i,k+1}, \dots, z_{i,k+K_p})$, shape $(K_p, 256)$ [35, 59].
*   **Trainable?** No direct gradients are propagated through this branch [38].
*   **Frozen?** No. Although its gradients are blocked, its weights $\bar{\theta}$ are updated at each training step using an **Exponential Moving Average (EMA)** of the Context Encoder's parameters $\theta$ [37, 38]:
    $$\bar{\theta} \leftarrow \eta\bar{\theta} + (1 - \eta)\theta$$
    where the decay rate is set to $\eta = 0.99$ [Table II].
*   **Paper gives exact architecture?** Yes. It mirrors the Context Encoder’s architecture exactly and shares identical parameter values at initialization [38].
*   **Need assumptions?** Same ResNet assumptions as the Context Encoder.

---

## 5. Masking

### **Role in the System**
This represents a key architectural departure. **TS-JEPA does not apply spatial/patch masking to the frame itself.** Instead, its "masking" is purely temporal: future frames are withheld from the Context Encoder and must be predicted by the Predictor [14, 33]. 

*   **Input shape?** N/A (no physical mask is applied to the frame).
*   **Output shape?** N/A.
*   **Trainable?** No.
*   **Frozen?** Yes.
*   **Paper gives exact architecture?** Yes. The paper explicitly contrasts TS-JEPA against standard visual JEPAs (like I-JEPA and V-JEPA) in Table I and Section III-A, highlighting that it does not use spatial masking but focuses on action-conditioned future prediction [14, 33].

---

## 6. Predictor ($P_phi$)

### **Role in the System**
The neural network deployed on the server. It captures the nonlinear physical dynamics of the system in latent space, predicting future embeddings based on the current embedding and a planned action sequence [36].

*   **Input shape?**
    *   **Autoregressive step (Standard inference loop)**: The concatenation of the previous embedding $z_{i,k}$ shape $(256,)$ and the normalized control force command $u_{i,k}$ shape $(1,)$, making a total input vector of shape $(257,)$ [36].
    *   **Multi-step formulation (Block inference)**: Current embedding $z_{i,k}$ shape $(256,)$ and sequence of predicted/planned control commands $(\tilde{u}_{i,k}, \dots, \tilde{u}_{i,k+K_p-1})$ shape $(15, 1)$ [36].
*   **Output shape?**
    *   **Autoregressive step**: Next predicted embedding $\tilde{z}_{i,k+1}$ of shape $(256,)$ [36].
    *   **Multi-step block**: Predicted sequence of shape $(15, 256)$ [36].
*   **Trainable?** Yes. It is parameterized by a set $\phi$ of learnable parameters [36].
*   **Frozen?** No. It is actively updated using gradient descent during TS-JEPA training [37]. (Frozen during the Semantic Actor training and run-time inference phases) [41, 44].
*   **Paper gives exact architecture?** Yes. It is structured as an **MLP with a single hidden layer of 1024 neurons and an output layer of 256 neurons** [59].
*   **Need assumptions?** Yes. We must assume the activation function for the hidden layer (typically ReLU, matching the encoder/actor) and the concatenation mechanics for the autoregressive feedback loop.

---

## 7. Latent Prediction Loss

### **Role in the System**
The loss function that computes alignment in the latent space. It forces the predictor to output embeddings that are semantically aligned with the target encoder's outputs [37].

*   **Input shape?** Predicted embeddings $(\tilde{z}_{i,k+1}, \dots, \tilde{z}_{i,k+K_p})$ of shape $(K_p, 256)$ and target embeddings $(z_{i,k+1}, \dots, z_{i,k+K_p})$ of shape $(K_p, 256)$ [36].
*   **Output shape?** Scalar float.
*   **Trainable?** No. This is a non-parametric mathematical metric.
*   **Frozen?** Yes.
*   **Paper gives exact architecture?** Yes, in Equation (13) [37]:
    $$\mathcal{L}_{\\text{TS-JEPA}} = \arg \min_{\theta, \phi} \frac{1}{K_s}\sum_{k=1}^{K_s} \frac{\langle \tilde{z}_{i,k+1}, z_{i,k+1} \rangle}{\|\tilde{z}_{i,k+1}\|_2 \cdot \|z_{i,k+1}\|_2}$$

> ### ⚠️ **CRITICAL REPRODUCTION BUG WARNING (Silent Failure Mode)**
> **Equation (13) Typo**: Taken literally, minimizing the raw cosine similarity will actively train the network to align predicted and target embeddings in **opposite directions** (pushing similarity toward $-1.0$). This is almost certainly a formatting or extraction typo in the PDF.
> 
> *   **The Fix**: You must minimize $1 - \text{cosine\_similarity}$ (which is equivalent to maximizing the cosine similarity).
> *   **PyTorch Implementation**:
>     ```python
>     # Correct loss formulation to maximize alignment
>     loss = 1.0 - torch.nn.functional.cosine_similarity(z_pred, z_target, dim=-1).mean()
>     ```
> *   *Coding the raw argmin directly will cause the model's loss curves to falsely appear as if it is "training" while actually driving the model toward maximum latent misalignment!*

---

## 8. Semantic Actor ($C_\varepsilon$)

### **Role in the System**
The control network deployed on the server. It bridges the latent representation and the physical environment by mapping semantic embeddings directly to horizontal physical force commands without pixel reconstruction [42, 43].

*   **Input shape?** Low-dimensional semantic embedding $z_{i,k}$ (or predicted embedding $\tilde{z}_{i,k}$) of shape $(256,)$ [43, 59].
*   **Output shape?** Predicted control command force $\tilde{u}_{i,k}$ of shape $(1,)$ [44, 56].
*   **Trainable?** Yes. Parameterized by a set $\varepsilon$ of learnable parameters [43].
*   **Frozen?** No. It is trained during its dedicated training phase after TS-JEPA training is finalized [44]. (Frozen during the run-time inference phase) [44].
*   **Paper gives exact architecture?** Yes. It is implemented as an **MLP with two hidden layers containing 1024 and 256 neurons respectively, each followed by an ReLU activation function** [59].
*   **Need assumptions?** Yes. We assume the output layer uses a continuous activation function (like Linear or Tanh) to scale the outputs back to the continuous physical range of $[-20\text{ N}, 20\text{ N}]$ [56].

---

## 9. Training

### **Role in the System**
The offline optimization pipeline used to train both TS-JEPA and the Semantic Actor sequentially.

*   **Input shape?**
    *   **TS-JEPA**: Dataset $D_s = \{x_{i,k}, u_{i,k}\}_{k=1}^{K_s}$ of consecutive state frames and normalized commands [39].
    *   **Semantic Actor**: Dataset $D_a = \{z_{i,k}, u_{i,k}\}_{k=1}^{K_a}$ of consecutive semantic embeddings and commands [43].
*   **Output shape?** Fully trained model weights ($\theta, \phi, \varepsilon$).
*   **Trainable?** Yes. This is the active optimization stage.
*   **Frozen?** Yes, the Target Encoder gradients are explicitly blocked through its branch during TS-JEPA training [38].
*   **Paper gives exact architecture?** Yes. Tables II and III provide the exact hyperparameters:
    *   **TS-JEPA**: SGD Optimizer, learning rate $= 2 \times 10^{-1}$ (decays by 0.99 every 20 epochs), batch size $= 256$, epochs $= 150$, weight decay $= 0.0004$, EMA decay rate $\eta = 0.99$ [58, Table II].
    *   **Semantic Actor**: AdamW Optimizer, learning rate $= 6 \times 10^{-3}$, batch size $= 200$, epochs $= 300$, dropout $= 0.2$, with early stopping enabled [58, Table III].

> ### ⚠️ **IMPLEMENTATION INSTABILITY WARNING (Learning Rate Paradox)**
> **Hyperparameter Discrepancy**: The TS-JEPA learning rate in Table II ($2 \times 10^{-1} = 0.2$) is unusually high for SGD on a ResNet + MLP regression task. By comparison, the Semantic Actor's LR is a more standard $6 \times 10^{-3}$ ($0.006$).
> 
> *   **The Risk**: This high learning rate could easily lead to training instability, exploding gradients, or early `NaN` returns during optimization.
> *   **Debugging Guideline**: If your local reproduction diverges, overflows, or returns `NaN` in the first few epochs, **immediately drop this learning rate** to a more conventional SGD learning rate (e.g., $1 \times 10^{-2}$ or $1 \times 10^{-3}$) before altering any other architectural properties.

---

## 10. Inference

### **Role in the System**
The deployed run-time execution phase. The local device compresses visual states to save bandwidth, and the server computes control commands, utilizing the predictor to handle network dropouts.

*   **Input shape?**
    *   **On-Device**: Raw visual sensor state $x_{i,k}$.
    *   **On-Server**: Uplink packet containing the semantic embedding $z_{i,k}$ shape $(256,)$ if the device is scheduled ($\alpha_{i,k} = 1$). If the packet is dropped/lost, the server receives nothing.
*   **Output shape?** Control command force $u_{i,k}$ of shape $(1,)$ sent to the actuator [56].
*   **Trainable?** No.
*   **Frozen?** Yes. All models ($\Psi_\theta, P_\phi, C_\varepsilon$) are frozen [41, 44].
*   **Paper gives exact architecture?** Yes. The execution loop is explicitly defined in Sections III-A and III-B:
    1.  If the packet is successfully received, the Actor maps $z_{i,k} \to \tilde{u}_{i,k}$ [44].
    2.  If the packet is lost, the Predictor runs autoregressively:
        $$\tilde{z}_{i,k+1} = P_{\phi}(\tilde{z}_{i,k} \ || \ \tilde{u}_{i,k})$$
        and the Semantic Actor computes the control forces directly from these predicted embeddings $\tilde{z}_{i,k}$ [41, 44].
*   **Need assumptions?** Yes. We assume that during prolonged blackouts exceeding the 15-step horizon, the system implements a fallback safety mechanism (e.g., applying $0\text{ N}$ force or locking the brakes) to prevent unbounded divergence.
