## 1. GPU and CPU Memory and Transfer Rates

### 1.1 GPU Memory

GPU memory, often called HBM (High Bandwidth Memory) or VRAM, is high bandwidth memory physically attached to the GPU package. It is designed to feed thousands of parallel compute cores efficiently.

Key characteristics:

- Very high bandwidth
- Low access latency relative to CPU memory
- Limited capacity compared to system RAM

Typical values for NVIDIA A100:

- Capacity: 40GB or 80GB HBM2e
- Peak bandwidth: ~1.6 TB/s

GPU memory stores:

- Model weights
- Activations
- Gradients
- Optimizer states
- KV cache during inference and decoding

---

### 1.2 CPU Memory

CPU memory refers to system RAM, typically DDR4 or DDR5, located on the motherboard.

Key characteristics:

- Much larger capacity
- Significantly lower bandwidth
- Higher latency compared to GPU memory

Typical values:

- Capacity: 64GB to 1TB+
- Bandwidth: ~50 to 100 GB/s per socket

CPU memory is commonly used for:

- Data loading and preprocessing
- Checkpoint storage before transfer
- Offloaded parameters or optimizer states in memory constrained setups

---

### 1.3 GPU to CPU Transfer Rates

Data movement between GPU and CPU happens over interconnects.

Approximate peak transfer rates:

- PCIe Gen4: ~32 GB/s
- PCIe Gen5: ~64 GB/s
- NVLink (A100): ~300 GB/s

Even with NVLink, transfer bandwidth is far lower than on-device GPU memory bandwidth, making frequent transfers expensive.

---

---

## 2. Bandwidth, Latency, and Compute

### 2.1 Bandwidth

Bandwidth measures how much data can be transferred per unit time, typically in GB/s or TB/s.

In training and inference:

- Large tensors are streamed repeatedly
- Sustained bandwidth determines throughput
- Many transformer operations are memory bandwidth bound

---

### 2.2 Latency

Latency is the delay to access the first byte of data.

- GPU memory latency is lower than CPU memory
- PCIe transfers have high latency relative to on-device access

Latency matters for:

- Small tensor operations
- Kernel launch overhead
- Synchronization points

For large matrix operations, bandwidth dominates over latency.

---

### 2.3 Compute

Compute refers to the raw arithmetic capability of the processor, measured in FLOPs.

A100 peak performance:
- FP16 or BF16 Tensor Core: ~312 TFLOPs

In practice:
- Many LLM workloads are not compute bound
- Compute units often wait on memory due to limited data reuse

---

### 2.4 Compute vs Memory Bound Regimes

- Compute bound: performance limited by FLOPs
- Memory bound: performance limited by data movement

Transformers often become memory bound during:

- Attention
- Layer normalization
- Optimizer updates during training

---

---

## 3. Training Considerations for a 7B Model on a Single A100

### 3.1 Memory Components During Training

Training requires storing four major components in GPU memory:

1. Model weights
2. Gradients
3. Optimizer states
4. Activations

We quantify each component explicitly.

#### Parameter Definitions

Let:

- $P = 7 \times 10^9$ parameters  
- BF16 precision for weights and gradients: 2 bytes  
- FP32 precision for optimizer states: 4 bytes  

---

#### 3.1.1 Model Weights

$$
M_{\text{weights}} = P \times 2 \text{ bytes}
$$

$$
= 7 \times 10^9 \times 2 = 14 \text{ GB}
$$

---

#### 3.1.2 Gradients

Each trainable parameter produces one gradient tensor.

$$
M_{\text{grads}} = P \times 2 \text{ bytes}
$$

$$
= 14 \text{ GB}
$$

Running total so far:

$$
28 \text{ GB}
$$

---

#### 3.1.3 Optimizer States (Adam or AdamW)

Adam maintains two FP32 states per parameter:

- First moment
- Second moment

$$
M_{\text{optimizer}} = P \times 2 \times 4 \text{ bytes}
$$

$$
= 7 \times 10^9 \times 8 = 56 \text{ GB}
$$

Running total:

$$
14 + 14 + 56 = 84 \text{ GB}
$$

This already exceeds the memory of an A100 80GB.

---

#### 3.1.4 Activations

Activation memory depends on:

- Number of layers $L$
- Batch size $B$
- Sequence length $T$
- Hidden dimension $H$

A simplified scaling relation:

$$
M_{\text{act}} = O(L \times B \times T \times H \times 2 \text{ bytes})
$$

For a 7B transformer with gradient checkpointing:

- Activations typically consume 5 to 15 GB

---

#### 3.1.5 Total Training Memory

$$
M_{\text{total}} = M_{\text{weights}} + M_{\text{grads}} + M_{\text{optimizer}} + M_{\text{act}}
$$

$$
\approx 90 \text{ to } 100 \text{ GB}
$$

This explains why full fine-tuning of a 7B model does not fit on a single A100.

---

### 3.2 Techniques That Enable Feasible Training

Common strategies include:

- Parameter-efficient fine-tuning such as LoRA or adapters
- Mixed precision training using BF16
- Gradient checkpointing to reduce activation memory
- Small micro-batches with gradient accumulation
- Limiting sequence length

These techniques primarily reduce:

- Gradient memory
- Optimizer state memory
- Activation memory

---

### 3.3 CPU Offloading

CPU offloading can move:

- Optimizer states
- Parameters

Tradeoffs:

- Enables fitting larger models
- Severely reduces training throughput
- Often impractical for production training

---

---

## 4. Memory Differences Between Training and Inference

### 4.1 Training Memory Profile

Training requires:

- Weights
- Activations from forward pass
- Gradients from backward pass
- Optimizer states

Memory usage is dominated by optimizer states and activations.

---

### 4.2 Inference Memory Profile

Inference requires:

- Model weights
- Activations for current forward pass
- KV cache for attention

Notably absent:

- Gradients
- Optimizer states

This is why inference fits models that training cannot.

---

### 4.3 Quantitative Difference

For a 7B model:

- Training memory: ~90GB or more
- Inference memory: ~20 to 30GB depending on sequence length and KV cache size

---

---

## 5. KV Cache Memory Usage

### 5.1 What Is the KV Cache

The KV cache stores the key and value tensors from the attention mechanism for previously processed tokens.

It avoids recomputing attention over the full context during autoregressive decoding.

---

### 5.2 Which Memory Does KV Cache Use

KV cache resides in:

- GPU memory during standard inference and serving
- CPU memory only in specialized offloading or paging setups

During high throughput inference:

- KV cache must stay in GPU memory
- Frequent access makes CPU storage impractical

---

### 5.3 KV Cache Memory Scaling

KV cache memory scales with:

- Number of layers
- Number of attention heads
- Sequence length
- Batch size

This makes KV cache the dominant memory consumer during long context inference.

---

---

## 6. Summary

Key takeaways:
- GPU memory is the primary constraint for both training and inference
- Bandwidth often limits performance more than compute
- Training requires additional memory for gradients and optimizer states
- Inference is cheaper because it avoids optimizer and gradient storage
- KV cache lives in GPU memory and dominates long-context inference memory

Understanding these tradeoffs is essential for designing and debugging large-scale LLM systems.
