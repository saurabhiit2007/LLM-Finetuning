# 🧩 LORA: Low-Rank Adaptation

---

### 1. Overview

Large Language Models (LLMs) contain billions of parameters, making full fine-tuning computationally expensive and memory intensive.  

**Low-Rank Adaptation (LoRA)** provides a **parameter-efficient** way to adapt pretrained models by freezing the original weights and introducing small trainable **low-rank update matrices**.  

LoRA decomposes weight updates into a **low-rank factorization**, allowing fine-tuning with only a fraction of the original parameters while retaining model quality.

---

### 2. Motivation

Fine-tuning a pretrained model requires adjusting all parameters, which can be:

- **Expensive** — requires large GPU memory and long training time.
- **Inefficient** — multiple downstream tasks need separate full fine-tunes.
- **Redundant** — many weight updates lie in a low intrinsic dimension subspace.

LoRA aims to address these issues by **restricting weight updates to a low-rank subspace**.

---

### 3. Core Idea

Let $W_0 \in \mathbb{R}^{d \times k}$ be a pretrained weight matrix of a layer (e.g., in attention or MLP).  
In full fine-tuning, the model learns a weight update $\Delta W$, resulting in:

$$
W = W_0 + \Delta W
$$

LoRA assumes $\Delta W$ is **low-rank** and can be decomposed as:

$$
\Delta W = B A
$$

where:

- $A \in \mathbb{R}^{r \times k}$
- $B \in \mathbb{R}^{d \times r}$
- $r \ll \min(d, k)$ is the **rank** hyperparameter.

During fine-tuning:

- $W_0$ is **frozen** (no gradient updates).
- Only $A$ and $B$ are **trainable**.

At inference, the effective weight is:

$$
W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A
$$

where $\alpha$ is a scaling factor controlling the magnitude of updates.

---

### 4. LoRA in Attention Layers

In Transformer architectures, LoRA is typically applied to **query (Q)** and **value (V)** projection matrices within the self-attention module.

For example, the modified query projection becomes:

\[
h = (W_Q + \Delta W_Q) x = W_Q x + B_Q A_Q x
\]

This retains the original computation while enabling efficient adaptation with small additional matrices.

---

### 5. Objective Function

LoRA uses the **same loss function** as the base fine-tuning objective (e.g., cross-entropy for language modeling):

\[
\mathcal{L} = - \sum_{t} \log p_\theta(y_t | y_{<t}, x)
\]

The only difference is that **only** the parameters in \( A \) and \( B \) are updated:

\[
\frac{\partial \mathcal{L}}{\partial W_0} = 0, \quad
\frac{\partial \mathcal{L}}{\partial A}, \frac{\partial \mathcal{L}}{\partial B} \neq 0
\]

This selective gradient flow drastically reduces training cost and memory footprint.

---

### 6. Implementation Details (Pseudo-Code)

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.A = nn.Parameter(torch.empty(r, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, r))
        
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        self.weight.requires_grad = False  # Freeze base weights

    def forward(self, x):
        return F.linear(x, self.weight + self.scaling * self.B @ self.A)
```


### 7. Hyperparameters & Heuristics

| **Hyperparameter** | **Typical Range** | **Practical Tip** |
|---------------------|------------------|-------------------|
| **Rank (r)** | 4 – 64 (sometimes up to 256) | Start small (4/8/16) and increase if underfitting |
| **Alpha (α)** | ≈ 2 × r | Scaling factor: `scaling = α / r` |
| **Learning Rate** | 1e-4 – 5e-4 | Too high → drift; too low → slow adaptation |
| **Dropout (`lora_dropout`)** | 0.0 – 0.1 | 0.05 often helpful on small datasets |
| **Epochs** | 1 – few | Avoid many epochs on small instruction datasets |

### 8. Training Configurations & Memory Optimizations

- **Mixed precision**: Use `fp16` or `bf16` to reduce memory usage and speed up training.  
- **Gradient accumulation**: Emulate large batch sizes using smaller per-device batches.  
- **Gradient checkpointing**: Trade compute for reduced activation memory footprint.  
- **CPU offload / `device_map`**: Offload frozen weights using the `accelerate` or Hugging Face `device_map` feature.  
- **Optimizer**: `AdamW` is the default; for very large adapter parameter sets, consider memory-efficient optimizers or even `SGD` if appropriate.  
- **QLoRA**: Load the base model in **4-bit** precision using `bitsandbytes`, and train LoRA adapters — enables **single-GPU training** for very large models.  


### 9. Common Issues and Concrete Solutions

#### 🧠 OOM / CUDA Out of Memory
- Lower `rank (r)`.  
- Use **QLoRA (4-bit)** or **mixed precision**.  
- Reduce **batch size** and use **gradient accumulation**.  
- Enable **gradient checkpointing** or **CPU offload**.  

---

#### ⚡ Training Instability / Divergence
- Lower `learning rate` and/or `α`.  
- Add a small **LoRA dropout**.  
- Use **warmup** and **learning rate schedulers** (e.g., cosine or linear).  

---

#### 🪫 Underfitting (Insufficient Capacity)
- Gradually increase **rank (r)**.  
- Add adapters to more modules (e.g., **MLP layers**).  

---

#### 🧩 Overfitting on Small Datasets
- Reduce **epochs** and **learning rate**.  
- Add **dropout** and **data augmentation**.  
- Use **early stopping** and **validation checks**.  

---

#### ⚙️ Quantization Compatibility Issues
- Prefer tested stacks: `bitsandbytes` + **Hugging Face** + `peft`.  
- Validate numeric stability on a small subset before full training.  

---

#### 🔗 Adapter Conflicts When Stacking
- Avoid overlapping **target modules** unless intentionally merging adapters.  
- Use explicit **adapter fusion tools** when combining multiple adapters.

---

### 10. Best Practices & Checklist

- Start with small **rank** `r = 4–16` and `α = 2 × r`.  
- **Freeze base model weights**; train only adapter parameters.  
- Use **mixed precision** and **gradient checkpointing** where appropriate.  
- Use **PEFT / Hugging Face tooling** for reliable save/load and metadata management.  
- Monitor **validation metrics** and **KL-like drift metrics** (compare outputs to base).  
- If memory constrained, use **QLoRA + LoRA adapters**.  
- Keep **logs, seeds, and repeat runs** for reproducibility.  

---

### 11. Limitations & Challenges

- **Rank–Capacity Tradeoff**: Small `r` may underfit; large `r` increases memory use and instability.  
- **Task-Specific Sensitivity**: Optimal values for `r`, `α`, and learning rate vary across models and tasks.  
- **Quantization Effects**: Combining LoRA with quantization (as in QLoRA) requires additional tuning.  
- **Adapter Management**: Multiple adapters need clear naming and metadata to avoid conflicts.  
- **Not a Universal Replacement**: For extreme distribution shifts, full fine-tuning may still be necessary.  

---

### 12. Comparison: LoRA vs Other Methods

| **Method**          | **Parameter Efficiency** | **Compute Cost** | **Flexibility** | **Notes** |
|----------------------|--------------------------|------------------|-----------------|------------|
| **Full fine-tuning** | ❌                      | High             | Moderate        | Updates all parameters |
| **Adapter tuning**   | ✅                      | Medium           | High            | Bottleneck MLPs per layer |
| **Prefix tuning**    | ✅                      | Low              | Medium          | Learned prompt vectors |
| **LoRA**             | ✅                      | Low              | High            | Mergeable, simple low-rank updates |
| **QLoRA**            | ✅✅                     | Very Low         | High            | 4-bit quantization + LoRA |

---