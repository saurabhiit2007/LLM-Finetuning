# Flash Attention

### 1. Overview

FlashAttention is a fast and memory-efficient implementation of the attention mechanism used in Transformer models. This repository explains what FlashAttention is, why it is faster than standard attention, and how it works under the hood, with a focus on interview preparation and practical understanding.

---

### 2. Motivation

Attention is the core operation behind Transformers, but standard attention becomes a major bottleneck for long sequences. The main problem is not only compute, but **memory movement**, which is often the true limiter on modern GPUs.

FlashAttention was introduced to:

- Reduce memory usage  
- Minimize expensive GPU memory reads and writes  
- Scale efficiently to long sequences  

---

### 3. Standard Attention and Its Limitations

Given query, key, and value matrices:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

While simple and elegant, this formulation has serious performance and memory issues for long sequences.

#### 3.1 Quadratic Memory Growth

Assume:

- Sequence length \(N = 16{,}384\)  
- FP16 precision (2 bytes per element)  

The attention score matrix \(QK^T\) has:

$$
N^2 = 16{,}384^2 \approx 268 \text{ million elements}
$$

Memory required just for the attention matrix:

$$
268\text{M} \times 2 \text{ bytes} \approx 512 \text{ MB}
$$

This does not include the softmax output, gradients during training, or activations from other layers, which can easily exceed GPU memory limits.

---

#### 3.2 Excessive Memory Traffic

Standard attention performs multiple memory-heavy steps:

1. Compute $QK^T$ and write to GPU global memory  
2. Read $QK^T$ back to apply softmax  
3. Write softmax output back to memory  
4. Read softmax output again to compute weighted sum with \(V\)  

Even with fast compute, repeated **global memory reads and writes** dominate runtime, making GPUs often memory-bound rather than compute-bound.

---

#### 3.3 Inefficient for Long Sequences (Code Example)

A simplified PyTorch-style implementation:

```python
import torch
import math

# Q, K, V shape: (batch, seq_len, num_heads, head_dim)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
attn = torch.softmax(scores, dim=-1)
output = torch.matmul(attn, V)
```

What happens internally:

- scores materializes a full $N×N$ tensor
- attn creates another $N×N$ tensor
- Both tensors live in global memory

As N grows, memory usage and latency grow quadratically.

#### 3.4 Numerical Issues with Low Precision

With FP16 or BF16:

- Large dot products in $QK^T$ can overflow
- Small values can underflow to zero

Standard attention often requires casting to FP32 for stability, which further increases memory usage and slows execution.

### 4. What Is FlashAttention and How It Works

FlashAttention is an **exact, memory-efficient attention algorithm**. It computes the same result as standard attention but avoids materializing the full $N \times N$ attention matrix. This makes it much faster and reduces GPU memory usage, especially for long sequences.

Key advantages:

- Handles long sequences efficiently (e.g., 4k+ tokens)  
- Works in FP16 and BF16 without numerical issues  
- Reduces memory bandwidth usage with minimal extra compute  

FlashAttention achieves this through three main ideas: **tiling**, **fused computation**, and **single-pass attention with online softmax**.

---

#### 4.1 Tiling

Instead of computing attention for the full sequence at once, FlashAttention splits the query, key, and value matrices into **small tiles** that fit into GPU shared memory.

**Example:**

- Sequence length: $N = 16{,}384$
- Tile size: $B = 128$  

Memory usage for a tile: $128 \times 128 = 16{,}384$ elements (much smaller than $(16{,}384)^2$)  

**Code-style intuition:**

```python
# pseudo-code for tiling
for q_tile in Q_tiles:
    for k_tile, v_tile in zip(K_tiles, V_tiles):
        partial_scores = q_tile @ k_tile.T
        # accumulate results incrementally

```

Benefit: Only a small block is in memory at a time, reducing GPU memory footprint dramatically.

#### 4.2 Fused Computation

FlashAttention fuses multiple steps into a single kernel:

1. Matrix multiplication $(Q \cdot K^T)$  
2. Scaling by $(1/\sqrt{d})$  
3. Softmax computation  
4. Weighted sum with $(V)$  

**Why this matters:**  

- Standard attention performs each step separately, writing intermediate results to global memory.  
- FlashAttention keeps all intermediate computations **in shared memory**, avoiding costly reads/writes.

**Example intuition:**

```python
# pseudo-code for fused attention
output_tile = flash_attention(q_tile, k_tile, v_tile)
```

Here, flash_attention does all four steps at once, producing the final output for that tile.

#### 4.3 Single-Pass Attention and Online Softmax

FlashAttention computes attention in one streaming pass:

- Compute partial scores for each tile
- Update running maximum and normalization term for softmax
- Accumulate output incrementally

This allows numerically stable softmax in FP16/BF16 without ever storing the full attention matrix.

Example numerical intuition:

- Tile 1 contributes scores [0.1, 0.5, 0.3]
- Tile 2 contributes [0.2, 0.4, 0.1]
- Running softmax computes the final normalized weights across tiles incrementally

Benefit:

- Exact same result as full attention
- Avoids overflow/underflow in low precision
- Reduces memory reads/writes drastically

#### 4.4 Practical Impact

- Memory complexity reduced from $O(N^2) → O(N⋅B)$ where $B$ is tile size
- Enables training with longer sequences or larger batch sizes
- Provides 2–4x speedups for long sequences on modern GPUs

Code example using PyTorch API:
```python
from flash_attn import flash_attn_func

# q, k, v shape: (batch, seq_len, num_heads, head_dim)
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
```

This produces exact attention results while being faster and more memory-efficient than standard attention.

### 5. When FlashAttention Helps (and When It Does Not)
Works best when:

- Sequence length is large (typically 2k tokens or more)
- Using FP16 or BF16
- Running on modern NVIDIA GPUs with fast shared memory

Less useful when:

- Sequence length is very short
- CPU-based inference
- Custom attention patterns not supported by FlashAttention kernels

### 6. Why is online softmax needed?

#### 6.1. Numerical Stability Problem

Standard softmax is computed as:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**Issue in FP16/BF16:**

- FP16 has limited precision (~3–4 decimal digits) and a small exponent range.  
- Large values of $(x_i)$ (e.g., 50) cause \(e^{x_i}\) to **overflow**.  
- Very negative values of $(x_i)$ (e.g., -50) cause $(e^{x_i})$ to **underflow** to zero.  
- Long sequences exacerbate the problem because summing hundreds or thousands of exponentials increases the risk of overflow/underflow.  

Without precautions, computing softmax in FP16 can produce **NaNs or zeros**, breaking both training and inference.

#### 6.2. Why “Online” Softmax Helps

FlashAttention computes attention **tile by tile**, so it cannot store the full $N \times N$ attention matrix. To compute softmax correctly across the entire sequence in FP16/BF16, it uses **online softmax**.

#### How It Works

1. Maintain a **running maximum** $m$ across tiles.
    - Shift scores before exponentiating: $e^{x_i - m}$
    - Prevents overflow in exponential.

2. Maintain a **running sum** of exponentials across tiles.
    - Partial sums from each tile are combined incrementally.  
    - Ensures correct normalization for the softmax over the full sequence.

3. Compute the weighted sum with $V$ **incrementally**.
    - No full softmax matrix is stored in memory.  
    - Output is accumulated as each tile is processed.

---

#### Example

Suppose we have **2 tiles** with attention scores:

- Tile 1: `[0.1, 0.5, 0.3]`  
- Tile 2: `[0.2, 0.4, 0.1]`

**Standard softmax** (if we could store all scores):

$$
\text{softmax}([0.1, 0.5, 0.3, 0.2, 0.4, 0.1])
$$

**Online softmax computation:**

1. **Tile 1**  
    - Running max \(m = 0.5\)  
    - Compute shifted exponentials: `[exp(0.1-0.5), exp(0.5-0.5), exp(0.3-0.5)] ≈ [0.67, 1.0, 0.82]`  
    - Running sum \(s = 0.67 + 1.0 + 0.82 = 2.49\)  
    - Partial weighted sum with $V$ stored in output

2. **Tile 2**  
    - New max $m = \max(0.5, 0.4) = 0.5$ (same in this case)  
    - Shifted exponentials: `[exp(0.2-0.5), exp(0.4-0.5), exp(0.1-0.5)] ≈ [0.74, 0.90, 0.61]`  
    - Update running sum: \(s = 2.49 + 0.74 + 0.90 + 0.61 = 4.74\)  
    - Accumulate weighted sum with $V$

3. **Normalization**  
    - Each accumulated output is divided by the final sum \(s = 4.74\)  
    - Produces **exact same softmax result** as computing on the full sequence

---

#### Key Benefits

- Computes **exact attention** even in FP16/BF16  
- Works efficiently with **long sequences** and **large tiles**  
- Avoids storing huge intermediate matrices  
- Reduces GPU memory usage and memory bandwidth overhead

> In short: Online softmax allows FlashAttention to compute attention tile by tile while staying numerically stable and memory-efficient.


### 7. End-to-End FlashAttention Example

Suppose we have:

- Sequence length $N = 8$ (small for simplicity)  
- Head dimension $d = 2$  
- Tile size $B = 4$  

We want to compute attention for a single head:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

---

#### Step 1: Prepare Q, K, V

```python
import torch
import math

Q = torch.tensor([[0.1, 0.2],
                  [0.3, 0.1],
                  [0.0, 0.4],
                  [0.5, 0.2],
                  [0.3, 0.3],
                  [0.1, 0.5],
                  [0.4, 0.0],
                  [0.2, 0.1]])  # shape: (8, 2)

K = Q.clone()  # for simplicity
V = torch.arange(8*2).reshape(8,2).float()  # dummy value matrix
```

#### Step 2: Split into Tiles

To reduce memory usage, FlashAttention splits the sequence into smaller **tiles** that fit into GPU shared memory.

- Tile size \(B=4\) → 2 tiles along the sequence  

```python
# Split Q, K, V into tiles
Q_tiles = [Q[:4], Q[4:]]  # tile 1 and tile 2
K_tiles = [K[:4], K[4:]]
V_tiles = [V[:4], V[4:]]
```

Benefit: Only a small portion of the sequence is in memory at a time, avoiding the need to materialize the full attention matrix.

#### Step 3: Process Tile 1

1. Compute partial scores in shared memory:

    $$
    \text{scores} = Q_\text{tile1} \cdot K_\text{tile1}^T / \sqrt{d}
    $$

    ```python
    scores_tile1 = Q_tiles[0] @ K_tiles[0].T / math.sqrt(2)
    ```

2. Apply online softmax:
    - Compute max of scores: m = scores_tile1.max(dim=1)
    - Shift and exponentiate: exp_scores = torch.exp(scores_tile1 - m)
    - Running sum: s = exp_scores.sum(dim=1)
    - Partial weighted sum with V: output_tile1 = (exp_scores @ V_tiles[0]) / s

Memory benefit: only a 4×4 matrix exists at a time.

#### Step 4: Process Tile 2 Incrementally

- Compute partial scores of Q_tile1 × K_tile2^T
- Update running max and running sum for online softmax
- Accumulate weighted outputs with V_tile2
- Repeat for Q_tile2 × K_tile1^T and Q_tile2 × K_tile2^T

No full 8×8 attention matrix is ever materialized.

#### Step 5: Accumulate Output

- Incrementally compute the weighted sum across all tiles
- Resulting output shape (8, 2) matches standard attention
- Softmax computed exactly using online normalization