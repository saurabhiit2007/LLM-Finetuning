## 1. Motivation

The Adam optimizer maintains additional state for each trainable parameter, which becomes a major memory bottleneck when training or fine tuning large language models.

For each parameter, Adam stores:

- First moment estimate $m$ in FP32
- Second moment estimate $v$ in FP32

This requires **8 bytes per parameter**, often exceeding available GPU memory even when model weights fit.

Paged Adam is designed to address this optimizer state memory bottleneck.

---

---

## 2. What Is Paged Adam

Paged Adam is a **memory managed variant of the Adam optimizer** that stores optimizer states in **CPU memory** and moves them to **GPU memory in small chunks only when needed** for the update step.

The Adam update rule itself is unchanged. The optimization focuses entirely on **where optimizer states live and how they are accessed**.

Paged Adam is most commonly implemented in the `bitsandbytes` library and is widely used in large model fine tuning workflows.

---

---

## 3. Core Idea

The key idea behind Paged Adam is inspired by virtual memory systems.

- Optimizer states reside primarily in host memory
- Small pages of optimizer states are transferred to the GPU
- Updates are applied for that page
- Updated states are written back to CPU memory
- The next page is processed

At no point do all optimizer states need to be resident in GPU memory.

---

---

## 4. Step by Step Operation

During each optimizer step:

1. Gradients are computed on the GPU.
2. A page of optimizer states $m, v$ is transferred from CPU memory to GPU memory.
3. The Adam update is applied for the corresponding parameters.
4. Updated optimizer states are transferred back to CPU memory.
5. The process repeats for the next page.

Only a small fraction of optimizer state memory is present on the GPU at any time.

---

---

## 5. Memory Benefits

Paged Adam dramatically reduces GPU memory usage by optimizer states.

- GPU memory usage becomes nearly constant with respect to model size
- Enables fine-tuning models that exceed GPU memory limits
- Particularly effective when combined with LoRA or QLoRA

This makes single GPU fine tuning of large models feasible.

---

---

## 6. Performance Tradeoffs

The main cost of Paged Adam is performance.

- Frequent CPU to GPU memory transfers
- PCIe or NVLink bandwidth becomes the bottleneck
- Optimizer step latency increases significantly

As a result:

- Training throughput is lower
- Paged Adam is better suited for fine-tuning than large scale pretraining

---

---

## 7. Relationship to Other Memory Saving Techniques

### 7.1 Paged Adam vs ZeRO Offload

| Aspect | Paged Adam | ZeRO Offload |
|------|------------|--------------|
| Scope | Optimizer states only | Parameters, gradients, optimizer |
| Granularity | Page level | Tensor or state level |
| Typical use | Single GPU setups | Multi GPU distributed training |
| Complexity | Relatively simple | Distributed system complexity |

---

### 7.2 Paged Adam and 8-bit Optimizers

Paged Adam is often combined with:

- 8-bit or 4-bit optimizer states

This reduces:

- CPU memory usage
- Data transfer volume per page

This combination is common in QLoRA training pipelines.

---

---

## 8. When to Use Paged Adam

Paged Adam is well suited for:

- Single GPU fine-tuning
- Memory constrained environments
- LoRA or QLoRA based training

Paged Adam is less suitable for:

- High throughput multi GPU training
- Full model pretraining
- Latency sensitive workloads

---

---

## 9. One Sentence Interview Summary

Paged Adam is a memory efficient variant of Adam that keeps optimizer states in CPU memory and pages them into GPU memory only during the update step, trading training throughput for the ability to fine tune models that would not otherwise fit in GPU memory.
