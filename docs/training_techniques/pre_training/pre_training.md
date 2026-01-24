## 1. Overview

Pre-training defines the *raw capability ceiling* of a Large Language Model. Architectural choices, data quality, and scaling decisions made at this stage dominate downstream performance far more than post-training alignment or prompting tricks. Most limitations observed later are traceable to decisions made here.

---

## 2. Objectives and Scaling

### 2.1 Self-Supervised Learning

LLMs are trained using **self-supervision**, where labels are derived directly from the data itself. Given a sequence of tokens:

$$
x = (x_1, x_2, \dots, x_T)
$$

the model learns to predict the next token:

$$
P(x_t \mid x_1, \dots, x_{t-1})
$$

This formulation:

- Requires no human annotation
- Scales naturally with data size
- Supports emergent behaviors such as reasoning and in-context learning

Despite its simplicity, this objective implicitly captures syntax, semantics, world knowledge, and procedural patterns.

---

### 2.2 Negative Log Likelihood Objective

Training minimizes the **Negative Log Likelihood (NLL)**:

$$
\mathcal{L} = - \mathbb{E}_{x \sim \mathcal{D}} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

Key properties:

- Equivalent to minimizing cross-entropy
- Strongly penalizes confident incorrect predictions
- Encourages calibration under idealized assumptions

Important limitation:

- NLL optimizes *average token prediction*, not task success, reasoning correctness, or truthfulness.

---

### 2.3 Chinchilla Scaling Laws (Training Optimality)

The Chinchilla results (Hoffmann et al.) showed that many prior models were **over-parameterized and under-trained**.

Core insight:
> For a fixed compute budget, model performance is maximized when **model size and training tokens are scaled proportionally** (roughly 20 tokens per parameter).

This shifted industry practice from "bigger models" to "compute-optimal training" with massive, high-quality corpora.

---

### 2.4 Inference-Optimal Scaling (The LLaMA Paradigm)

#### 2.4.1 Background: Why Chinchilla Is Not the Full Story

The **Chinchilla scaling laws** optimize for *training compute efficiency*.  
They answer the question:

> Given a fixed training compute budget, how should we allocate it between model size and training tokens to minimize loss?

This framing is correct for research experiments and one-off model training runs.  
However, it ignores a critical real-world constraint:

**Most of the cost of an LLM is paid after training, during inference.**

In production systems, a model may be:

- Trained once
- Served millions or billions of times

This changes the optimization target entirely.

#### 2.4.2 Training Cost vs Inference Cost

It is essential to distinguish the two:

**Training cost**

- Scales with: parameters × tokens × optimization overhead
- Paid once

**Inference cost**

- Scales primarily with: number of active parameters
- Paid per request
- Dominates total cost in deployed systems

Key insight:
> Inference cost depends on *model size*, not on how many tokens the model saw during training.

This means that Chinchilla-optimal models may be cheap to train but expensive to serve.


#### 2.4.3 The Shift in Optimization Objective

This leads to a new question:

> Given a fixed inference budget or deployment footprint, how do we maximize model quality?

This is where **inference-optimal scaling** emerges.

#### 2.4.4 Chinchilla-Optimal vs Inference-Optimal

#### Chinchilla-Optimal Scaling

- Train very large models
- Use relatively fewer training tokens
- Minimizes training loss per unit of training compute

This results in:

- Large parameter counts
- High inference latency and cost
- Practical difficulty deploying at scale

Modern open-weights models (e.g., LLaMA 3) are "over-trained" by Chinchilla standards (e.g., >100x tokens per parameter) to maximize performance for a specific deployment footprint.

#### Inference-Optimal Scaling

- Train smaller models
- Train them on *far more tokens* than Chinchilla recommends
- Accept higher training cost to reduce inference cost

This results in:

- Fewer parameters
- Lower latency and memory usage
- Better cost-performance trade-off in production

#### 2.4.5 Why Over-Training Smaller Models Works

Smaller models are **capacity-limited**, not data-limited.

By exposing them to vastly more data:

- Representations become more robust
- Rare patterns are reinforced
- Generalization improves significantly

Even though Chinchilla would label this as "over-training", the additional data:

- Continues to reduce downstream error
- Improves reasoning and instruction-following
- Makes the model competitive with much larger alternatives

#### 2.4.6 The LLaMA Paradigm

Modern open-weight models such as **LLaMA 2 and LLaMA 3** follow this strategy.

Characteristics:

- Moderate parameter count
- Extremely high tokens-per-parameter ratio
- Often tens to hundreds of times more tokens per parameter than Chinchilla-optimal

This design choice:

- Maximizes quality for a fixed inference budget
- Enables efficient deployment on limited hardware
- Makes open models more practical for real-world use

#### 2.4.7 Learnings

#### 1. Scaling Laws Are Objective-Dependent
There is no single "optimal" scaling rule.
- Chinchilla optimizes training compute
- Inference-optimal scaling optimizes deployment cost

Always ask: *What is the objective being optimized?*

#### 2. Training Longer Can Be Better Than Training Bigger
For production systems:
- Smaller, heavily trained models often outperform
- Larger, lightly trained models are harder to serve

This reverses earlier intuitions from the pre-Chinchilla era.

#### 3. Tokens Are Cheaper Than Parameters at Inference Time
- Extra training tokens increase one-time cost
- Extra parameters increase recurring cost

In large-scale deployments, recurring costs dominate.

#### 4. Model Design Is a Systems Problem
Model size, data scale, and deployment constraints are tightly coupled.
Good LLM design requires:
- ML theory
- Systems thinking
- Cost-aware decision making

---

#### 2.4.8 Compute vs Data vs Parameters

Pre-training is a three-way trade-off:

| Resource | Bottleneck Effect |
|--------|------------------|
| Parameters | Capacity ceiling |
| Data | Generalization and robustness |
| Compute | Training duration and batch size |

Common failure modes:

- Too many parameters leads to memorization
- Too little data leads to brittle generalization
- Insufficient compute leads to undertrained representations

Modern frontier models are primarily **data-limited**, not architecture-limited.

---

## 3. Data Pipeline and Quality

Data quality is often more important than model size. "Garbage in, garbage out" applies strictly to LLMs.

### 3.1 Raw Data Sources

Typical pre-training corpora include:

- Web crawl data (CommonCrawl)
- Books and long-form text
- Code repositories (GitHub)
- Mathematical and scientific text (arXiv)
- Structured and semi-structured documents

Each domain contributes different inductive biases:

- Code improves logical consistency and state tracking
- Math improves symbolic manipulation
- Long-form text improves discourse modeling

### 3.2 Data Cleaning and PII Redaction

Cleaning steps typically include:

- **HTML boilerplate removal:** Strips navigation bars, scripts, ads, and markup artifacts so the model learns from meaningful textual content rather than page structure noise.
- **Unicode normalization:** Converts visually or semantically equivalent characters into a canonical form to reduce vocabulary fragmentation and stabilize token statistics.
- **Language-specific token normalization:** Applies rules tailored to each language, such as lowercasing, diacritic handling, or script normalization, to improve consistency and learning efficiency.
- Removal of corrupted or truncated documents

**PII Redaction (Privacy):**

Strictly required for enterprise models to prevent regurgitating private data.

- **Regex-based:** Removing emails, SSNs, phone numbers.
- **Entity Recognition:** Replacing proper names with placeholders (e.g., `<PERSON>`).
- **Memorization Audits:** Checking if the model can generate unique sequences from the training set.

---

### 3.3 Exact and Near-Duplicate Removal

Duplicates distort the training distribution and waste compute.

Techniques:

- **Exact hashing:** SHA-256 matching.
- **MinHash / LSH:** Locality-sensitive hashing for near-duplicates.
- **Embedding-based similarity:** For semantic duplicates.

Why this matters:

- Inflated frequency biases
- Artificially low validation loss
- Memorization instead of abstraction

---

### 3.4 Leakage and Benchmark Contamination

Leakage sources:

- Public benchmark solutions in web data
- GitHub repositories with answers
- Fine-tuning data overlapping with evaluation sets

Consequences:

- Inflated benchmark scores
- Misleading claims of reasoning ability
- Poor real-world generalization

Mitigation requires proactive filtering (n-gram matching against test sets) and post-hoc auditing.

---

### 3.5 Data Mixing and Annealing

How different data sources are combined determines the model's flavor.

- **Static Mixing:** Pre-assigned weights (e.g., 60% Web, 20% Code, 10% Math).
- **Dynamic Selection (DoReMi):** Continuously adjusts the sampling weights of different data sources during training using feedback from a smaller proxy model, prioritizing data that most improves validation loss and downweighting less useful or noisy sources.
- **Annealing:** A form of curriculum learning where high-quality data (synthetic, textbooks, math) is upsampled heavily in the final 5-10% of training to "polish" the model's skills.

---

### 3.6 Synthetic Data Generation and Risks

Synthetic data is increasingly used to:

- Fill data gaps
- Emphasize rare skills
- Bootstrap reasoning behaviors

However, risks include:

- **Model Collapse:** Reinforcement of model errors leading to distribution narrowing.
- Reduced diversity and creativity.

Uncontrolled synthetic feedback loops can permanently damage model quality.

---

## 4. Architecture Choices

### 4.1 Decoder-Only Transformers

Most LLMs use a **decoder-only Transformer** due to:

- Causal attention alignment with autoregressive training
- Simpler deployment and KV caching
- Strong scaling behavior

---

### 4.2 Mixture of Experts (MoE)

MoE is the standard for scaling model capacity while keeping inference cheap.

- **Concept:** Feed-Forward Network (FFN) layers are split into multiple "experts."
- **Routing:** A learnable gate selects only top-$k$ experts (usually $k=2$) for each token.
- **Sparse Activation:** A model might have huge *total* parameters (e.g., 8x7B) but low *active* parameters per token.
- **Trade-offs:** High capacity and low latency, but training can be unstable (load balancing) and memory bandwidth heavy.

---

### 4.3 Modern Component Standards (The "No-Vanilla" Transformer)

Standard Transformers (2017) are rarely used. Modern defaults include:

- **RMSNorm:** Replaces LayerNorm. It is computationally simpler (no mean centering) and numerically stable.
- **Pre-Norm:** Normalization is applied *before* attention/FFN layers (improves gradient flow) rather than after.
- **SwiGLU:** An activation function that replaces ReLU/GeLU. It adds a gating mechanism, increasing parameters slightly but improving convergence significantly.
- **Bias-Free Layers:** Removing bias terms from Linear layers to improve stability.

---

### 4.4 Tokenization

Common approaches:

- BPE (Byte Pair Encoding)
- SentencePiece Unigram

**Modern Nuances:**

- **Byte-Fallback:** Falls back to raw bytes for unknown characters to ensure no `<UNK>` tokens exist (crucial for code).
- **Digit Splitting:** Splitting numbers (e.g., "2025" $\to$ "2", "0", "2", "5") rather than grouping them improves arithmetic reasoning.
- **Trade-offs:** Larger vocabularies compress text better (faster inference) but increase the embedding layer size and training difficulty.

---

### 4.5 Positional Embeddings

Modern models use **Rotary Positional Embeddings (RoPE)**.

Advantages:

- Implicit relative positioning
- Better extrapolation to longer contexts
- Compatible with attention optimizations (FlashAttention)

---

### 4.6 Attention Variants

To reduce memory and compute:

- **Multi-Query Attention (MQA):** All heads share one KV head.
- **Grouped-Query Attention (GQA):** Compromise where groups of heads share a KV head (Standard in LLaMA).

Benefits:

- Drastically lower KV cache memory usage
- Faster inference decoding

---

### 4.7 Context Length Scaling

Increasing context length impacts:

- Memory quadratically (without FlashAttention/Ring Attention)
- Training stability (loss spikes)

Common strategies:

- **Long-context fine-tuning:** Pre-train on short context (e.g., 4k), then anneal on long context (e.g., 128k).
- **Ring Attention:** For training on sequences longer than single-GPU memory.

---