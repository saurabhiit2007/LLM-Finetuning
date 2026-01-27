## 1. Overview

**vLLM** is an open-source inference engine optimized for high-throughput, low-latency serving of large language models. Its core innovation, **PagedAttention**, rethinks KV cache management by applying principles similar to virtual memory systems in operating systems.

The key contribution of vLLM is not faster kernels alone, but **dramatically improved GPU memory utilization**, enabling higher batch sizes, better multi-tenancy, and more predictable latency under load.

---

---

## 2. The KV Cache Bottleneck in LLM Inference

### 2.1 Why the KV Cache Dominates
During autoregressive decoding, each generated token appends **Key and Value tensors** for every transformer layer. Over long sequences or many concurrent requests, the KV cache quickly becomes the dominant consumer of GPU memory.

Key properties:
- Grows linearly with sequence length
- Must be retained across decoding steps
- Is memory-bandwidth bound during decode

---

---

## 3. The Core Technology: PagedAttention

The KV cache (the memory storing previous tokens to predict the next one) is the primary bottleneck in scaling LLMs.

### The Problem: Fragmentation

Standard inference engines allocate a contiguous "max-length" chunk of memory for every request.

* **Over-reservation:** Reserving 2048 tokens for a request that only generates 50.
* **Internal Fragmentation:** Memory wasted inside the reserved block.
* **Waste:** Up to 60-80% of VRAM is often left unused but "reserved."

### The Solution: PagedAttention

vLLM breaks the KV cache into fixed-size **blocks** (pages).

* **Logical Blocks:** Sequential tokens in the prompt.
* **Physical Blocks:** Non-contiguous memory addresses on the GPU.
* **Block Table:** A mapping system that allows the model to access these blocks as if they were one continuous string.

**Result:** Waste is reduced to **<4%**, allowing for significantly larger batch sizes.

---

---

## 4. Scheduling: Continuous Batching

Traditional engines use "Static Batching," where the entire batch must finish before new requests start.

* **Iteration-Level Scheduling:** vLLM schedules at the level of individual iterations. 
* **Mechanism:** As soon as one sequence in a batch hits an `<EOS>` (End of Sentence) token, a new request from the queue is inserted into its spot in the next iteration.
* **Outcome:** Eliminates "bubbles" (idle time) in GPU utilization.

---

---

## 5. Modern (2025-2026) Advanced Features

### A. Speculative Decoding
vLLM implements speculative decoding where a **smaller draft model** (e.g., a 100M parameter model) predicts several tokens, and a **larger target model** (e.g., Llama 3 70B) verifies them in a single pass.
* **Benefit:** Reduces latency by 2-3x for heavy models.

### B. Automatic Prefix Caching (APC)
For RAG or multi-turn chat, vLLM caches the KV blocks of common prefixes (like system prompts).
* If two users share the same 1,000-token system prompt, vLLM stores it **once** in physical memory, and both requests point to the same blocks.

### C. Multi-LoRA Support
vLLM can serve one base model with hundreds of different fine-tuned "adapters" (LoRAs) simultaneously. 
* It uses specialized **SGMV (Shrink-Generalized Matrix-Vector)** kernels to compute multiple different LoRAs in a single batch without a significant performance hit.

---

---

## 6. Prefill vs Decode: Performance Characteristics

| Phase | Characteristics | Bottleneck |
| --- | --- | --- |
| Prefill | Parallel over tokens | Compute-bound |
| Decode | Sequential token-by-token | Memory-bandwidth-bound |

vLLM introduces **Chunked Prefill** to prevent long prompts from blocking decode for other users.

---

---

## 8. Memory Pressure and Preemption

When GPU memory becomes constrained, vLLM supports **preemption** strategies:

- **Swap:** Move KV blocks to CPU memory
- **Recompute:** Drop KV blocks and recompute them later

Recompute trades extra compute for lower memory pressure and is often preferred on fast GPUs.

---

---

## 9. Architectural Comparison

| Feature | vLLM | Hugging Face TGI | NVIDIA TensorRT-LLM |
| :--- | :--- | :--- | :--- |
| **Memory Mgmt** | PagedAttention | FlashAttention | Paged KV Cache |
| **Ease of Use** | High (Pythonic) | Medium (Rust/Go) | Low (Complex Build) |
| **Best For** | General Throughput | Stability/HF ecosystem | Peak NVIDIA Perf |

---

---

## 10. Useful Memory Approximation

Approximate KV cache memory usage in bytes:

$$
\text{KV Memory} \;\approx\; 2 \times L \times T \times H \times D_h \times B
$$


Where:
- 2 accounts for **Keys and Values**
- $L$ is the number of layers  
- $H$ is the number of attention heads 
- $B$ is the number of bytes per element (2 for FP16, 1 for FP8)
- $D_h$ is the per-head hidden dimension
- $T$ is the sequence length (number of cached tokens)

**Example:**

For **Llama-3 8B**:

- $L = 32$
- $D_{\text{model}} = 4096$
- $B = 2$ (FP16)

Per token KV cache memory:

$$
2 \times 32 \times 4096 \times 2
\approx 524{,}288 \text{ bytes} \approx 0.5 \text{ MB}
$$

Approximate totals:
- 2k tokens → ~1 GB KV cache
- 8k tokens → ~4 GB KV cache

This linear scaling with sequence length explains why **KV cache memory becomes the dominant bottleneck** and motivates techniques such as **PagedAttention** in vLLM.

---

---

## Q & As

**Q1: How does vLLM handle a situation where the GPU runs out of memory during a request?**
* **A:** It uses **Preemption**. It can either "Swap" (move blocks to CPU RAM) or "Recompute" (drop the blocks and re-calculate them later when memory is free).

**Q2: What is the difference between the 'Prefill' and 'Decode' phases?**
* **A:** **Prefill** processes the input prompt (parallel/compute-bound). **Decode** generates tokens one by one (sequential/memory-bound). vLLM uses **Chunked Prefill** to prevent large prompts from stalling the generation of other users.

**Q3: Why is vLLM better for multi-tenant SaaS?**
* **A:** Because of PagedAttention and Multi-LoRA support. It allows hosting many different "specialized" models on a single GPU cluster with minimal overhead.

---

