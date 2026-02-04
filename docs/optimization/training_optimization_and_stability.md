Training large neural networks, especially Transformers, requires careful choices of optimizers, learning rate schedules, and numerical techniques to ensure fast convergence, stable training, and good generalization. This section covers the most important concepts commonly discussed in interviews and used in practice.

---

## 1. Optimizers and Schedules

### 1.1 Adam vs AdamW

#### Adam Optimizer
Adam (Adaptive Moment Estimation) is one of the most widely used optimizers in deep learning.

**Key ideas:**

- Maintains an exponential moving average of gradients (first moment)
- Maintains an exponential moving average of squared gradients (second moment)
- Uses bias correction for both moments
- Adapts the learning rate per parameter

**Update rule (simplified):**

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Where:**

- $\theta_t$: Model parameters at training step $t$.
- $\theta_{t+1}$: Updated model parameters after applying one optimization step.
- $\eta$ (Learning Rate): Global step size that controls how large the parameter update is.
- $\hat{m}_t$ (Bias Corrected First Moment): Exponentially decayed moving average of past gradients, corrected for initialization bias.  
  Represents the estimated mean of the gradients.
  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
  $$
- $\hat{v}_t$ (Bias Corrected Second Moment): Exponentially decayed moving average of squared gradients, corrected for initialization bias.  
  Represents the estimated variance of the gradients.
  $$
  \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$
- $\beta_1$: Exponential decay rate for the first moment estimate.  
  Typical value: 0.9
- $\beta_2$: Exponential decay rate for the second moment estimate.  
  Typical value: 0.999
- $\epsilon$: Small constant added for numerical stability to prevent division by zero.  
  Typical value: $10^{-8}$

**Advantages:**

- Fast convergence
- Works well with sparse gradients
- Requires minimal tuning compared to SGD

**Limitations:**

- Implicitly couples L2 regularization with adaptive learning rates
- Often leads to worse generalization compared to SGD or AdamW in large models

---

#### AdamW Optimizer
AdamW decouples weight decay from the gradient-based update.

**Key difference from Adam:**

- Weight decay is applied directly to parameters, not via the gradient

**Update rule (conceptually):**

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
$$

**where**:

- $\lambda$ (Weight Decay Coefficient): Controls the strength of weight decay regularization.  
  Larger values enforce stronger penalization of large parameter magnitudes.
- $- \eta \lambda \theta_t$ (Decoupled Weight Decay Term): Applies weight decay directly to the parameters, independent of the gradient-based update.  
  This ensures consistent regularization regardless of adaptive learning rates.


**Why AdamW matters:**

- Correctly implements weight decay as regularization
- Prevents adaptive learning rates from weakening regularization
- Standard optimizer for modern Transformers (BERT, GPT, ViT)

> Note: AdamW is almost always preferred over Adam for large-scale Transformer training.

---

### 1.2 Learning Rate Warmup and Decay

#### Learning Rate Warmup

**What it is:**

- Gradually increases the learning rate from a small value to the target value over the first few training steps

**Why it is needed:**

- Large models have unstable gradients at initialization
- Prevents divergence caused by large updates early in training
- Especially critical for Transformers and mixed precision training

**Common strategies:**

- Linear warmup
- Warmup for a fixed number of steps (e.g., 1k to 10k steps)

---

#### Learning Rate Decay

After warmup, the learning rate is gradually reduced to improve convergence.

**Common decay schedules:**

- Linear decay
- Cosine decay
- Step decay
- Polynomial decay

**Cosine decay (widely used):**

- Smoothly reduces learning rate
- Avoids sudden drops that can destabilize training

> Insight: Warmup handles early instability, decay improves late-stage convergence and generalization.

---

### 1.3 Weight Decay and Regularization

#### Weight Decay

- Penalizes large weights to prevent overfitting

**Important distinction:**

- L2 regularization modifies the loss
- Weight decay directly modifies the parameter update

**Why decoupling matters:**

- With adaptive optimizers, L2 regularization is not equivalent to weight decay
- AdamW fixes this mismatch

<details>
<summary><strong>Explanation: L2 regularization is not equivalent to weight decay</strong></summary>

1. L2 Regularization: It adds a penalty term to the loss function:

$$
\mathcal{L}' = \mathcal{L} + \frac{\lambda}{2} \|\theta\|^2
$$

Taking the gradient:

$$
\nabla_\theta \mathcal{L}' = \nabla_\theta \mathcal{L} + \lambda \theta
$$

So the optimizer update becomes:

$$
\theta_{t+1} = \theta_t - \eta \left( \nabla_\theta \mathcal{L} + \lambda \theta_t \right)
$$

This means regularization is applied through the gradient. <br><br>

2. Why This Works for SGD <br><br>

For SGD, the update is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L} - \eta \lambda \theta_t
$$

Here, L2 regularization is mathematically equivalent to weight decay because: <br>
- All parameters use the same learning rate <br>
- No per parameter scaling is applied <br><br>

So both methods shrink weights uniformly. <br><br>

3. What Breaks with Adam <br><br>

Adam modifies the update using adaptive, per parameter learning rates:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

When L2 regularization is added, the penalty term $\lambda \theta_t$ is also scaled by the adaptive denominator:

$$
\theta_{t+1} =
\theta_t -
\eta \frac{\hat{m}_t + \lambda \theta_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Consequence <br><br>

- Parameters with large gradient variance receive less regularization<br>
- Parameters with small gradient variance receive more regularization<br>
- Regularization strength becomes parameter dependent and inconsistent<br><br>

This is not true weight decay.<br><br>

4. What Weight Decay Actually Means<br><br>

True weight decay directly shrinks parameters independently of gradients:

$$
\theta_{t+1} = (1 - \eta \lambda)\theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Key properties:<br>

- Uniform shrinkage across parameters<br>
- Independent of gradient statistics<br>
- Matches the intended regularization behavior<br><br>

5. How AdamW Fixes the Problem <br><br>

AdamW explicitly decouples weight decay from the gradient update:

$$
\theta_{t+1} =
\theta_t -
\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} -
\eta \lambda \theta_t
$$
<br><br>

6. Why this works <br><br>

- Weight decay is applied directly to parameters<br>
- Adaptive learning rates affect only the gradient update<br>
- Regularization strength remains consistent<br><br>

</details>

#### Intuition Summary<br><br>

| Method | How Regularization Is Applied | Problem |
|------|-------------------------------|--------|
| SGD + L2 | Uniform parameter shrinkage | None |
| Adam + L2 | Scaled by adaptive learning rates | Inconsistent regularization |
| AdamW | Direct parameter decay | Correct behavior |


#### Other Regularization Techniques

- Dropout
- Label smoothing
- Data augmentation
- Early stopping

**Transformer-specific note:**

- Biases and LayerNorm parameters often exclude weight decay
- This is a common best practice in large models

---

## 2 Numerical Stability

### 2.1 FP16 vs BF16

#### FP16 (Half Precision)

**Characteristics:**

- 16-bit floating point
- Limited exponent range
- Higher risk of overflow and underflow

**Challenges:**

- Gradient underflow for small values
- Requires loss scaling for stability

---

#### BF16 (Brain Floating Point)

**Characteristics:**

- 16-bit floating point with larger exponent range
- Same exponent range as FP32
- Lower mantissa precision than FP16

**Advantages:**

- Much more numerically stable than FP16
- Usually does not require loss scaling
- Widely supported on TPUs and newer GPUs

> Insight: BF16 is preferred when hardware supports it due to better stability with minimal complexity.

---

### 2.2 Mixed Precision Training

This technique uses both FP16/BF16 and FP32 to get the "best of both worlds": the speed of 16-bit and the accuracy of 32-bit.

- **Forward Pass:** Done in FP16 for speed.
- **Loss Scaling:** Since FP16 has a narrow range, gradients can "underflow" (become zero). We multiply the loss by a large scale factor to push gradients into a representable range.
- **Master Weights:** A copy of the weights is kept in FP32. The gradients are converted to FP32 to update these master weights, ensuring precision isn't lost over time.


#### Benefits

- Faster training
- Lower memory usage
- Enables larger batch sizes and models

#### Loss Scaling

**Why it is needed (mainly for FP16):**

- Prevents gradients from underflowing to zero

**How it works:**

- Multiply loss by a scale factor before backprop
- Divide gradients by the same factor before optimizer step

**Dynamic loss scaling:**

- Automatically adjusts scale based on overflow detection
- Common in frameworks like PyTorch AMP

---

### 2.3 Gradient Clipping

#### What is Gradient Clipping?

To prevent "Exploding Gradients" (where a large update ruins the model weights), we cap the gradients.

- **Value Clipping:** Caps each element of the gradient at a min/max.
- **Norm Clipping:** Scales the entire gradient vector so its $L_2$ norm does not exceed a threshold. This preserves the direction of the gradient while limiting the magnitude.

Norm Clipping is more common.

$$
g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|}\right)
$$

where $\tau$ is the clipping threshold.

---

#### Why it matters

- Prevents exploding gradients
- Stabilizes training in deep or recurrent models
- Especially important for large learning rates or noisy gradients

**Typical values:**

- Global norm between 0.5 and 1.0 for Transformers

---

### 2.4 Loss Spikes and Divergence Diagnosis

#### Common Causes of Loss Spikes

- Learning rate too high
- Insufficient warmup
- Numerical overflow in FP16
- Poor initialization
- Data outliers or corrupted batches

---

#### Diagnosis Checklist

- Check learning rate schedule and warmup length
- Monitor gradient norms
- Enable gradient clipping
- Inspect loss scaling behavior
- Compare FP16 vs BF16 runs
- Verify data preprocessing and labels

---


#### Practical Debugging Tips

- Reduce learning rate and re-run
- Increase warmup steps
- Switch from FP16 to BF16 if possible
- Enable anomaly detection for NaNs and Infs
- Log per-layer gradient norms

---

## Summary

- AdamW is the default optimizer for modern large models
- Learning rate warmup is critical for early stability
- Weight decay must be decoupled from adaptive updates
- BF16 offers better numerical stability than FP16
- Mixed precision improves efficiency but requires care
- Gradient clipping and monitoring are essential debugging tools

These concepts form the backbone of stable and efficient training for large-scale neural networks and are frequently tested in machine learning interviews.
