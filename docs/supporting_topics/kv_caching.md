### 📦1. Self Attention Recap

Given hidden states $X \in \mathbb{R}^{T \times d}$:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Per head attention:

$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

Autoregressive decoding generates one token at a time with causal masking.

---

### 📦2. Why KV Cache Is Needed

At decoding step $t$, keys and values for tokens $1 \ldots t-1$ are unchanged but would be recomputed without caching.

This repeated computation dominates inference latency and wastes FLOPs.

---

### 📦3. KV Cache Mechanism

For each transformer layer $\ell$:

$$
\text{KVCache}_\ell = \{K_\ell^{1:t}, V_\ell^{1:t}\}
$$

At decoding step $t$:

- Compute $Q_t, K_t, V_t$
- Append $K_t, V_t$ to the cache
- Attend over all cached keys and values

$$
\text{Attn}_t = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_h}}\right)V_{1:t}
$$

Only the current token requires new computation.

---

### 📦4. Toy Example

Prompt: "I like neural"

Step 1: generate `"networks"`

- Compute and cache keys and values for the prompt
- Attend to all cached tokens

Step 2: generate `"models"`

- Reuse cached keys and values
- Compute keys and values only for `"networks"`

Previously generated tokens are never recomputed.

---

### 📦5. Complexity Analysis

#### Notation
- $T$: number of generated tokens
- $L$: number of transformer layers
- $H$: number of attention heads
- $d_h$: head dimension

#### Without KV Cache

At each decoding step, attention is recomputed for all previous tokens:

$$
O(L \cdot H \cdot d_h \cdot T^3)
$$

#### With KV Cache

Only attention against cached keys and values is computed:

$$
O(L \cdot H \cdot d_h \cdot T^2)
$$

KV caching removes one full factor of $T$ from decoding complexity.

---

### 📦6. Memory Cost

Each layer stores:

$$
K, V \in \mathbb{R}^{H \times T \times d_h}
$$

Total KV cache memory across all layers:

$$
O(L \cdot H \cdot T \cdot d_h)
$$

For long context inference, KV cache memory is often the dominant bottleneck.

### 📦7. Inference v/s Training Usage

#### 7.1 During Inference

This is the most common and important usage.

**Inference Workflow**

- Encode prompt
- Initialize empty KV cache per layer
- For each generated token:
    - Compute $Q_t, K_t, V_t$
    - Append $K_t, V_t$ to cache
- Compute attention using cached tensors

**Practical Benefits**

- Faster decoding
- Lower FLOPs
- Enables long context generation
- Essential for streaming and chat systems

#### 7.2 During Training

KV caching is not used in standard full sequence training.

**Why?**

- Training processes full sequences in parallel
- All tokens attend to each other simultaneously
- No repeated computation across steps

---

### 📦8. Scaling KV Cache for Long Context

Long context inference is primarily limited by KV cache memory, which grows linearly with sequence length.

#### 8.1 Sliding Window Attention

Only retain keys and values for the most recent $W$ tokens:

$$
K_{t-W:t}, V_{t-W:t}
$$

This bounds memory usage and is commonly used in streaming and chat applications. Older context is no longer directly accessible.

---

#### 8.2 KV Cache Quantization

KV cache quantization reduces memory usage and memory bandwidth by storing cached keys and values in lower precision formats. This is especially important for long context inference, where KV cache memory dominates total GPU usage.

#### What Gets Quantized

Both keys and values can be quantized, but they have different sensitivity:

- **Keys (K)** directly affect attention scores $QK^T$
- **Values (V)** affect the weighted sum after softmax

As a result:

- Keys usually require higher precision
- Values tolerate more aggressive quantization

---

#### Common Quantization Schemes

| Component | Typical Format | Notes |
|--------|----------------|------|
| Keys | FP16 / BF16 | Preserves attention score stability |
| Values | INT8 | Large memory reduction with minimal quality loss |
| Both | INT8 or INT4 | Used for extreme long context scenarios |

Mixed precision KV cache is widely used in practice.

#### Quantization Granularity

KV cache quantization can be applied at different levels:

- **Per tensor**: One scale for entire K or V tensor
- **Per head**: Separate scale per attention head
- **Per channel**: Separate scale per head dimension

Finer granularity improves accuracy but increases metadata and compute overhead.

#### Dequantization During Attention

At decoding step $t$:

1. Load quantized $K, V$ from cache
2. Dequantize to FP16 or BF16
3. Compute attention normally:

$$
\text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_h}}\right)V_{1:t}
$$

Dequantization cost is small compared to memory bandwidth savings.

#### Impact on Performance

Benefits:

- 2x to 4x KV memory reduction
- Higher batch size and longer context
- Improved inference throughput due to reduced memory traffic

Tradeoffs:

- Slight loss in generation quality
- Additional dequantization overhead

In practice, value quantization has minimal impact on quality, while aggressive key quantization requires careful tuning.

#### Interaction with Other Optimizations

- **GQA** further reduces KV cache size and works well with quantization
- **Paged KV cache** benefits from smaller KV blocks
- **FlashAttention** amortizes dequantization overhead inside fused kernels

---

#### 8.3 Prefix Caching

When multiple requests share a common prompt prefix, the KV cache for that prefix is computed once and reused across requests. This improves throughput in serving systems with templated prompts.

---

#### 8.4 Paged KV Cache

KV cache blocks can be moved between GPU and CPU or NVMe memory. This enables extremely long context lengths while trading off additional latency for cache paging.

---

### 📦9. Grouped Query Attention (GQA)

Grouped Query Attention reduces KV cache size by using fewer key value heads than query heads.

#### 9.1 Head Configuration

$$
H_q > H_k = H_v
$$

Example:

- Query heads $H_q = 32$
- Key value heads $H_k = 8$

This reduces KV cache memory by a factor of $H_q / H_k$.

#### 9.2 QK Computation with Mismatched Heads

Each key value head is shared by a fixed group of query heads.

Let:

$$
g = \frac{H_q}{H_k}
$$

Each key value head serves $g$ query heads.

For query head $i$, the corresponding key value head index is:

$$
\left\lfloor \frac{i}{g} \right\rfloor
$$

The attention computation becomes:

$$
\text{Attn}_i = \text{softmax}\left(\frac{Q_i K_{\left\lfloor i/g \right\rfloor}^T}{\sqrt{d_h}}\right)V_{\left\lfloor i/g \right\rfloor}
$$

Keys and values are reused directly without additional projection or averaging.

#### 9.3 Why GQA Is Effective

- Query heads retain expressive power
- Keys and values capture shared context
- KV cache size and memory bandwidth are significantly reduced

GQA is widely used in production LLMs.

---

### 📦10. Other Common Optimizations

#### FlashAttention

FlashAttention optimizes the attention kernel to reduce memory reads and improve numerical stability. It is complementary to KV caching and often used together.

#### Chunked Prefill

Long prompts are processed in chunks to incrementally build the KV cache. This avoids GPU out of memory errors during prefill.

#### Speculative Decoding

Both draft and target models maintain KV caches. When draft tokens are accepted, the target model reuses its cached keys and values, avoiding recomputation and increasing decoding throughput.
