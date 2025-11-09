# 🧩 Prefix Tuning

### 1. Overview

Large Language Models (LLMs) have billions of parameters, making full fine-tuning **computationally expensive** and **memory intensive**.

**Prefix Tuning** provides a **parameter-efficient** method to adapt pretrained models by keeping the model weights frozen and prepending **trainable continuous vectors (“prefixes”)** to the input of each Transformer layer.

The key idea is to learn task-specific prompts in the hidden space rather than modifying the model weights directly, allowing small memory footprint fine-tuning.

---

### 2. Motivation

Fine-tuning a pretrained model fully is often:

* **Expensive** — requires high GPU memory and long training cycles.
* **Inefficient** — multiple tasks need separate full fine-tunes.
* **Redundant** — only a small portion of the model’s hidden space needs adaptation for many tasks.

Prefix tuning addresses this by **modifying the activations at each layer via trainable prefix vectors**, keeping all original model parameters frozen.

---

### 3. Core Idea

Let a Transformer layer have hidden states $h \in \mathbb{R}^{L \times d}$, where:

* $L$ is the sequence length
* $d$ is the hidden dimension

Prefix tuning introduces **learnable prefix vectors** $P \in \mathbb{R}^{L_p \times d}$ (with $L_p \ll L$), prepended to the key and value projections of each attention layer:

$$
\tilde{K} = [P_K; K], \quad \tilde{V} = [P_V; V]
$$

where:

* $P_K, P_V \in \mathbb{R}^{L_p \times d_k}$ are trainable prefixes
* $K, V$ are original key and value matrices
* $[;]$ denotes concatenation along the sequence dimension

During training:

* **Original model weights are frozen**.
* Only the prefix vectors $P$ are **updated**.

At inference, the model uses:

$$
\text{Attention}(Q, \tilde{K}, \tilde{V})
$$

allowing adaptation without modifying the model weights.

---

### 4. Prefix Tuning in Attention Layers

In self-attention, standard attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

With prefix tuning:

* $K$ and $V$ are **augmented** with trainable prefix vectors:

$$
\tilde{K} = [P_K; K], \quad \tilde{V} = [P_V; V]
$$

* This injects **task-specific context** without changing $Q$, $K$, $V$ weights.
* The model effectively learns **a small task embedding** that guides attention.

---

### 5. Objective Function

Prefix tuning uses the **same loss as standard fine-tuning** (e.g., cross-entropy for language modeling):

$$
\mathcal{L} = - \sum_t \log p_\theta(y_t | y_{<t}, x)
$$

Gradients only flow through the prefix vectors:

$$
\frac{\partial \mathcal{L}}{\partial \text{model weights}} = 0, \quad
\frac{\partial \mathcal{L}}{\partial P} \neq 0
$$

This drastically reduces memory usage while maintaining strong task performance.

---

### 6. Implementation Details (Pseudo-Code)

```python
class PrefixTuning(nn.Module):
    def __init__(self, hidden_size, prefix_length=10, num_layers=12):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        # Trainable prefix vectors per layer
        self.prefix_keys = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, hidden_size))
            for _ in range(num_layers)
        ])
        self.prefix_values = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, hidden_size))
            for _ in range(num_layers)
        ])

    def forward(self, layer_idx, K, V):
        # Prepend trainable prefixes
        P_K = self.prefix_keys[layer_idx]
        P_V = self.prefix_values[layer_idx]
        K_aug = torch.cat([P_K, K], dim=0)
        V_aug = torch.cat([P_V, V], dim=0)
        return K_aug, V_aug
```

---

### 7. Hyperparameters & Heuristics

| **Hyperparameter**      | **Typical Range**       | **Practical Tip**                            |
| ----------------------- | ----------------------- | -------------------------------------------- |
| **Prefix length (L_p)** | 5 – 50                  | Longer prefixes for complex tasks            |
| **Hidden size**         | Match model hidden size | Usually same as Transformer hidden dimension |
| **Learning Rate**       | 1e-4 – 5e-4             | Small LR helps stable training               |
| **Dropout**             | 0.0 – 0.1               | Helps prevent overfitting                    |
| **Epochs**              | 1 – few                 | Avoid overfitting on small datasets          |

---

### 8. Training Configurations & Memory Optimizations

* **Mixed precision** (`fp16` / `bf16`) for memory and speed.
* **Gradient checkpointing** to save activation memory.
* **CPU offload** for frozen model weights using `accelerate` or `device_map`.
* **Optimizer**: `AdamW` or memory-efficient optimizers for prefix parameters.
* **Single-GPU friendly**: Prefix tuning only trains a small number of parameters.

---

### 9. Common Issues and Concrete Solutions

#### 🧠 OOM / CUDA Out of Memory

* Prefix tuning is already lightweight; reduce **prefix length** if necessary.
* Use **mixed precision** and **gradient checkpointing**.

#### ⚡ Training Instability

* Reduce **learning rate**.
* Add **dropout** to prefixes.
* Use learning rate schedulers or warmup.

#### 🪫 Underfitting

* Increase **prefix length**.
* Add prefixes to **more layers**.

#### 🧩 Overfitting on Small Datasets

* Reduce **epochs**.
* Add **dropout** or **early stopping**.

---

### 10. Best Practices & Checklist

* Start with **prefix length $L_p$ = 10–20**.
* **Freeze base model weights**; train only prefix vectors.
* Use **mixed precision** and **gradient checkpointing** where needed.
* Log validation metrics and monitor **task performance drift**.
* For large models, prefix tuning can be combined with **LoRA** or **QLoRA** for stronger adaptation.

---

### 11. Limitations & Challenges

* **Prefix length sensitivity**: Too short → underfitting; too long → memory overhead.
* **Layer selection**: Optimal layers for prefix insertion may vary.
* **Task generalization**: Prefixes are task-specific; transferring to new tasks may require re-training.
* **Not suitable for extreme distribution shifts**: Full fine-tuning may still be needed.

---

### 12. Comparison: Prefix Tuning vs Other Methods

| **Method**           | **Parameter Efficiency** | **Compute Cost** | **Flexibility** | **Notes**                                     |
| -------------------- | ------------------------ | ---------------- | --------------- | --------------------------------------------- |
| **Full fine-tuning** | ❌                        | High             | Moderate        | Updates all parameters                        |
| **Adapter tuning**   | ✅                        | Medium           | High            | Bottleneck MLPs per layer                     |
| **Prefix tuning**    | ✅                        | Low              | Medium          | Learned prefix vectors prepended to attention |
| **LoRA**             | ✅                        | Low              | High            | Mergeable, low-rank updates                   |
| **QLoRA**            | ✅✅                       | Very Low         | High            | 4-bit quantization + LoRA                     |

---
