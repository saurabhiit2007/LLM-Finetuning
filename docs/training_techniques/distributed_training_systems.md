This page covers the system architecture, parallelism strategies, and engineering trade offs required to train large language models at scale.

---

## 1. Data Parallelism (DP)

The most widely used baseline parallelism strategy where the full model is replicated across $N$ workers and each worker processes a different shard of the input batch.

### Core Mechanism

- Each GPU performs forward and backward passes on its local mini batch.
- Gradients are synchronized across all workers using All Reduce.
- After synchronization, every replica applies the same optimizer update, keeping parameters identical.

### PyTorch Distributed Data Parallel (DDP) Details

#### Gradient Bucketing

- Model parameters are grouped into **buckets** based on size (The default bucket size is about 25 MB, configurable via bucket_cap_mb).
- Instead of waiting for all gradients to be computed, DDP starts communication as soon as a bucket is ready.
- This reduces idle time by avoiding one large synchronization at the end of backward pass.

**Why this matters:**  
Smaller, incremental communication keeps GPUs busy and reduces the impact of network latency.

#### Asynchronous Gradient Reduction

- When gradients for a bucket are computed, DDP launches an **asynchronous All Reduce** operation.
- While communication is happening, the backward pass continues computing gradients for later layers.
- This overlaps computation and communication.

**Key insight:**  
Ideally, by the time backward computation finishes, most gradient communication is already done.

#### Synchronization Timing

- Gradients are synchronized **once per training iteration**, not per layer.
- Each parameter’s gradient is reduced exactly once after it is computed.
- The optimizer step is performed only after all gradient reductions complete.

**Common misconception:**  
DDP does not pause after every layer to synchronize. Synchronization is event driven and overlaps with backpropagation.


### **Limitations**
- Model parameters, gradients, and optimizer states must fit on a single GPU.
- Scaling is limited by global batch size and optimizer stability.
- Communication cost grows linearly with the number of GPUs.

---

## 2. Tensor Parallelism (TP)

Tensor Parallelism splits individual layers across devices so that a single layer computation is distributed.

### 2.1 Core Mechanism

Large weight matrices are partitioned across GPUs.

Consider a linear layer:

Y = XW

where:

- X has shape `[batch, hidden_in]`
- W has shape `[hidden_in, hidden_out]`
- Y has shape `[batch, hidden_out]`
  - Column parallel splits `W` by output dimension.
  - Row parallel splits `W` by input dimension.
- Each GPU computes a partial result which is combined using collective communication.

---

#### Column Parallelism (Output Dimension Split)

In column parallelism, the weight matrix W is split **by columns**:

W = [W₁ | W₂ | ... | Wₖ]

Each GPU holds:

- Wᵢ with shape `[hidden_in, hidden_out / k]`

Each GPU computes:

- Yᵢ = X · Wᵢ

At this point:

- Each Yᵢ is only a **partial output**.
- The full output Y is formed by concatenating all Yᵢ along the feature dimension.

**Communication Pattern**

- An All Gather is used to assemble the full Y across GPUs.
- This happens during the forward pass.
- During backpropagation, gradients w.r.t. X require an All Reduce.

**Key Insight**


Column parallelism parallelizes the **output features** and is commonly used in feed forward layers.

---

#### Row Parallelism (Input Dimension Split)

In row parallelism, the weight matrix W is split **by rows**:

W = [ W₁
      W₂
      ...
      Wₖ ]

Each GPU holds:

- Wᵢ with shape `[hidden_in / k, hidden_out]`

The input X is also split:

- X = [X₁ | X₂ | ... | Xₖ]

Each GPU computes:

- Yᵢ = Xᵢ · Wᵢ

Now:

- Each Yᵢ is a **partial sum** of the final output.

**Communication Pattern**

- An All Reduce is required to sum Yᵢ across GPUs.
- The final Y is identical on all GPUs after reduction.
- Backward pass mirrors this communication pattern.

**Key Insight**

Row parallelism parallelizes the **input features** and avoids an All Gather in the forward pass.

---

#### Why Two Parallelization Schemes Exist

Column and row parallelism are complementary:
- Column parallelism produces partial outputs that must be gathered.
- Row parallelism produces partial sums that must be reduced.

Modern transformer implementations alternate between them:
- Column parallel for linear projections.
- Row parallel for output projections.

This design minimizes total communication while keeping memory balanced.

---

### 2.2 Communication Characteristics

- Requires All Reduce or All Gather inside the forward and backward pass.
- Communication is frequent and latency sensitive.
- Performance depends heavily on interconnect bandwidth.

### Insights

**Strengths**

- Enables training when individual layers do not fit on a single GPU.
- Reduces per GPU activation and parameter memory.

**Constraints**

- Communication overhead is high.
- Usually restricted within a node due to bandwidth requirements.
- Sensitive to imbalance across tensor shards.

**Practical Usage**

- Often combined with Data Parallelism.
- Popularized by Megatron LM style architectures.

---

## 3. Pipeline Parallelism (PP)

Pipeline Parallelism splits the model by layer depth across devices.

### Core Mechanism

- Each GPU owns a contiguous block of layers.
- Activations flow forward through the pipeline.
- Gradients flow backward in reverse order.

### The Pipeline Bubble

- Without micro batching, only one stage is active at a time.
- GPUs sit idle waiting for inputs or gradients.

### Micro Batching

- The global batch is split into micro batches.
- Multiple micro batches are in flight simultaneously.
- Improves utilization at the cost of activation memory.

### Insights

**Strengths**

- Reduces per GPU parameter memory.
- Enables training extremely deep models.

**Trade-offs**

- Increased activation memory footprint.
- Pipeline schedule complexity.
- Backward pass latency increases.

**Systems Insight**

Pipeline parallelism improves memory scaling but hurts latency sensitive workloads.

---

## 4. ZeRO (Zero Redundancy Optimizer)

ZeRO is designed to **eliminate memory redundancy** in Data Parallel training. In standard DP, every GPU stores:

1. Model parameters
2. Gradients
3. Optimizer states (e.g., Adam’s momentum and variance)

This redundancy quickly becomes a bottleneck for very large models. ZeRO partitions these states across GPUs so that **each GPU only stores a fraction of the total memory**, enabling training of models that would otherwise not fit.

---

### ZeRO Stages

| Stage | Partitioned States | Memory Savings | Key Idea |
|------|-------------------|----------------|----------|
| Stage 1 | Optimizer states | ~4× | Each GPU keeps a shard of the optimizer states instead of a full copy. Gradients and parameters are still replicated. |
| Stage 2 | Gradients | ~8× | Gradients are also sharded. Each GPU contributes its shard to global All Reduce during backprop. |
| Stage 3 | Parameters | N× (number of GPUs) | Even the model parameters are partitioned. Each GPU only holds a subset and fetches other shards when needed for computation. |

**Example:**  
Suppose you have a 10B parameter model across 8 GPUs:
- Stage 1: Each GPU holds 1/1 of parameters but 1/8 of optimizer states.  
- Stage 2: Each GPU holds 1/8 of gradients.  
- Stage 3: Each GPU holds 1/8 of parameters, gradients, and optimizer states.  

---

### Key Insight

- Stage 1 and 2 mainly reduce memory replication.
- Stage 3 reduces the largest memory cost: the **parameters themselves**.
- At higher stages, computation must **fetch shards of parameters or gradients from other GPUs** on the fly.
- Communication becomes the primary bottleneck, requiring overlap with computation for efficiency.

---

### ZeRO Offload

ZeRO can also offload states to CPU RAM or even NVMe storage to free GPU memory:

- Only the portion of optimizer states or parameters **actively needed** resides on the GPU.
- Trades GPU memory pressure for **PCIe / NVMe bandwidth**.
- Makes training extremely large models possible even with limited GPU memory (e.g., 10B+ parameters on 4×A100 40GB).

**Practical Note:**  
Offloading is especially useful when the model is too big for Stage 3 alone or when using lower-end GPU clusters.

---

### Communication Patterns

- Stage 1: Minimal communication; only optimizer state shards are reduced.
- Stage 2: Requires All Reduce for gradient shards.
- Stage 3: Each forward/backward pass may require fetching remote parameter shards.
- Communication-computation overlap is critical to avoid GPU idling.

---

### Summary

- ZeRO scales memory linearly with the number of GPUs.
- Stage 3 maximizes memory saving but **communication overhead increases**.
- Poor network bandwidth or high latency can dominate runtime.
- Elastic / hybrid parallelism often combines ZeRO with DP, TP, or PP for large-scale training.

---

## 5. Communication and Memory Trade-offs

Distributed training is fundamentally constrained by a three way trade off between **memory capacity**, **compute throughput**, and **communication bandwidth**. Improving one dimension often degrades another, and real systems are designed to balance all three.

---

### 5.1 The Memory–Compute–Communication Triangle

- **Memory** limits how large a model and batch can fit on a single GPU.
- **Compute** determines how fast forward and backward passes can be executed.
- **Communication** determines how quickly GPUs can synchronize parameters, gradients, or activations.

For large models, memory is usually the first bottleneck. Techniques that reduce memory usage often increase either compute or communication cost.

### 5.2 Memory Accounting with Adam and FP16

Assume a model with `Ψ` parameters trained using Adam and mixed precision.

#### Parameter Storage

- Model weights stored in FP16 or BF16: `2Ψ` bytes

#### Gradient Storage

- Gradients stored in FP16 or BF16 during backpropagation: `2Ψ` bytes

#### Optimizer States

Adam maintains:

- FP32 master weights: `4Ψ` bytes  
- First moment (momentum): `4Ψ` bytes  
- Second moment (variance): `4Ψ` bytes  

Total optimizer memory:  
`12Ψ` bytes

#### Total Training Memory

Adding all components:

- Parameters: `2Ψ`
- Gradients: `2Ψ`
- Optimizer states: `12Ψ`

**Total:** approximately `16Ψ` bytes per parameter

---

#### Concrete Example

For a 7B parameter model:

- 7B × 16 bytes ≈ **112 GB**

This number excludes:

- Activation memory
- Temporary buffers
- Communication workspaces

This explains why even a single training replica of a 7B model cannot fit on an 80 GB GPU without memory optimization techniques.

---

### 5.3 Why Communication Becomes the Bottleneck

As memory saving techniques reduce per GPU state:

- More parameter shards must be fetched remotely.
- Gradients must be synchronized more frequently.
- Communication moves from being infrequent and bulk to frequent and latency sensitive.

High bandwidth interconnects and overlap with compute become critical.

---

### 5.4 Memory Reduction Techniques and Their Trade-offs

#### Activation Checkpointing

- Saves memory by discarding activations during forward pass.
- Recomputes activations during backward pass.
- Trades memory for additional compute, typically 20 to 40 percent overhead.

**When to use:**  

Memory constrained training where compute is not the bottleneck.

---

#### Mixed Precision Training

- Uses FP16 or BF16 for forward and backward computation.
- Keeps optimizer states in FP32 for numerical stability.
- Reduces memory usage and communication bandwidth.

**Key benefit:**  

Improves both memory efficiency and throughput with minimal accuracy loss.

---

#### Parameter Sharding

- Splits parameters, gradients, or optimizer states across GPUs.
- Removes redundancy present in Data Parallelism.
- Increases communication during forward and backward passes.

**Typical examples:**  

ZeRO Stage 2 and Stage 3.

> Takeaway: Large scale training is constrained by a memory–compute–communication trade off. Techniques like mixed precision, activation checkpointing, and parameter sharding reduce memory pressure but introduce additional compute or communication costs. Efficient systems overlap communication with computation to avoid performance collapse.

---

## 6. Checkpointing and Fault Tolerance

At large cluster scale, hardware and network failures are expected. Training systems must be designed to recover quickly with minimal loss of progress.

---

### 6.1 Checkpointing Strategies

**Full Checkpoints**

- Store model parameters, optimizer states, and RNG state.
- Enable exact training resumption.
- Expensive in terms of storage size and write time.

<details>

<summary> RNG State (Random Number Generator State) </summary>

1. Why RNG State Matters

Randomness is used in multiple parts of training:

- Weight initialization
- Dropout masks
- Data shuffling
- Data augmentation
- Stochastic layers or kernels

If training is resumed **without restoring RNG state**:

- Dropout patterns change
- Data order may differ
- Gradient noise changes

As a result, training can diverge from the original run, making debugging and reproducibility difficult.

2. What Is Typically Included

A full checkpoint usually stores RNG states for:

- Python `random`
- NumPy RNG
- PyTorch CPU RNG
- PyTorch CUDA RNG (per GPU)

In distributed training, each worker maintains its own RNG state, which must be saved and restored independently.

</details>

**Sharded Checkpoints**

- Each worker writes only its shard of parameters or optimizer state.
- Reduces checkpoint time and I/O contention.
- Requires coordinated restore logic.

**Asynchronous Checkpointing**

- Checkpointing runs in the background while training continues.
- Avoids blocking GPUs on slow storage.
- Slightly increases complexity and memory usage.

---

### 6.2 Fault Tolerance

**Elastic Training**

- Allows workers to join or leave during training.
- Automatically rebalances data and workloads.
- Commonly implemented using PyTorch Elastic or TorchRun.

**Health Checks**

- Detect hung, slow, or non communicating workers.
- Prevents a single faulty GPU from stalling the entire job.

**Straggler Detection**

- Identifies workers that are consistently slower.
- Helps avoid synchronization delays in collective operations.

---

> Takeaway: Checkpointing is fundamentally an I/O and systems problem. Efficient training requires minimizing checkpoint overhead while ensuring fast and correct recovery from failures.


---

## 7. Advanced Parallelism and Optimization Topics

These topics often differentiate strong systems candidates.

### 7.1 FlashAttention

- Computes attention in tiles to avoid materializing full attention matrices.
- Reduces memory from quadratic to linear in sequence length.
- Improves both speed and memory usage.

---

### 7.2 Mixture of Experts (MoE)

- Sparse activation where only a subset of parameters are used per token.
- Requires expert parallelism and routing strategies.
- Trades compute efficiency for model capacity.

---

### 7.3 Sequence Parallelism

Sequence parallelism splits the **sequence length dimension** of the input across multiple devices, rather than splitting model parameters or batch elements.

#### Core Idea

For an input tensor with shape:

`[batch, sequence_length, hidden_dim]`

- The sequence length dimension is partitioned across GPUs.
- Each GPU processes a contiguous chunk of tokens from the sequence.
- This reduces per GPU activation memory, which scales linearly with sequence length.

#### Why It Is Useful

- Attention and activation memory grow with sequence length.
- Very long context models (e.g., 32k, 64k, or 128k tokens) quickly exceed GPU memory limits.
- Sequence parallelism allows long sequences to fit by distributing token level computation.

#### Communication Pattern

- Certain operations, such as attention and layer normalization, require communication across sequence shards.
- Communication typically uses All Gather or All Reduce.
- Efficient overlap with computation is necessary to avoid performance loss.

#### Relationship with Tensor Parallelism

- Sequence parallelism is often combined with tensor parallelism.
- Tensor parallelism splits hidden dimensions, while sequence parallelism splits tokens.
- This combination balances memory usage and communication overhead.

> Takeaway: Sequence parallelism distributes tokens across GPUs to reduce activation memory for long context models, trading additional communication for the ability to train with very large sequence lengths.

---

### 7.4 Low Precision Training

- FP8 reduces bandwidth and memory.
- Requires careful scaling and error management.
- Hardware dependent and increasingly common on newer accelerators.

---

## 8. Additional Common Interview Topics

These are frequently asked in senior level ML systems interviews.

### 8.1 Hybrid Parallelism

- Real systems combine DP, TP, PP, and ZeRO.
- Interviewers often ask how to scale from 1 GPU to hundreds or thousands.
- A strong answer mentions hierarchical parallelism.

---

### 8.2 Throughput vs Latency

Throughput and latency represent two different optimization goals, and distributed systems make very different design choices depending on which one is prioritized.

---

#### Training: Throughput Oriented

- The primary goal in training is **maximum tokens processed per second**.
- Large batch sizes are used to fully utilize GPUs.
- Parallelism strategies favor efficiency even if individual requests are slow.

**Common choices in training**

- Data parallelism with large global batches.
- Pipeline parallelism with micro batching.
- Aggressive overlap of communication and computation.
- Higher tolerance for end to end latency per batch.

---

#### Inference: Latency Oriented

- The primary goal in inference is **fast response time per request**.
- Batch sizes are often small or dynamic.
- Minimizing synchronization and communication is critical.

**Common choices in inference**

- Replication rather than sharding for small models.
- Limited or no pipeline parallelism due to bubble overhead.
- Kernel fusion and caching over global synchronization.

---

#### Why Parallelism Strategies Differ

- Training can amortize communication over large batches.
- Inference cannot hide communication latency as easily.
- Techniques like tensor or pipeline parallelism may improve throughput but often increase per request latency.

> Takeaway: Training systems optimize for throughput using large batches and heavy parallelism, while inference systems optimize for low latency by minimizing synchronization and communication, leading to fundamentally different parallelism strategies.

--- 

### 8.3 Gradient Accumulation

Gradient accumulation is a technique used to simulate a larger batch size by accumulating gradients over multiple forward and backward passes before performing an optimizer update.

#### Core Idea

- Instead of updating model parameters after every mini batch, gradients are accumulated over `K` steps.
- The optimizer step is executed only once after all `K` gradients are accumulated.
- This creates an **effective batch size** of `K × mini_batch_size`.

---

#### Why It Saves Memory

- Each step processes a small mini batch that fits in GPU memory.
- Activations are freed after each backward pass.
- Only gradients are accumulated, avoiding the need to store a large batch at once.

---

#### Communication Benefits in Distributed Training

- In Data Parallel training, gradient synchronization normally happens every step.
- With gradient accumulation, synchronization happens only once every `K` steps.
- This reduces communication frequency and improves scalability.

---

#### Common Confusion with Data Parallelism

- Data parallelism distributes different data samples across GPUs.
- Gradient accumulation repeats multiple steps **on the same GPU** before synchronization.
- The two techniques are complementary and often used together.

---

#### Practical Considerations

- Learning rate schedules often need adjustment for large effective batch sizes.
- Loss values are usually scaled to avoid gradient magnitude changes.
- Very large accumulation steps can slow convergence.

> Takeaway: Gradient accumulation increases effective batch size by accumulating gradients across multiple steps, reducing memory usage and communication frequency, and is often combined with data parallelism to scale training efficiently.

---