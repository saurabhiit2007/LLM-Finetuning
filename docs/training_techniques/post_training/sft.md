Supervised Fine-Tuning (SFT) is the stage where a **pre-trained base model** is transformed into a **useful assistant** that follows instructions, respects formats, and exhibits desired interaction behavior. While pre-training builds a broad world model, SFT shapes *how* that knowledge is expressed.

From an interview perspective, SFT is best understood as **behavioral alignment via supervised learning**.

---

## 1. What SFT Optimizes

**Goal**  

Map broad knowledge into a consistent, controllable interface.

**Conceptual shift**

- Pre-training: “Continue the text”
- SFT: “Respond appropriately to a user instruction”

This is achieved without reinforcement learning. The training signal is fully supervised.

---

## 2. Instruction Tuning and Task Formatting

### Instruction Tuning

Instruction tuning teaches the model to condition its output on an explicit instruction rather than implicit continuation.

**Training objective**

Standard cross-entropy loss with **selective loss masking**.

If:

- $x$ = prompt tokens (system + user)
- $y$ = assistant response tokens

Then SFT minimizes:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{|y|} \log P(y_t \mid x, y_{<t})
$$

Tokens belonging to $x$ are excluded from the loss.

**Why masking matters**

- Prevents memorization of prompts
- Ensures gradients only optimize response generation
- Stabilizes alignment behavior

---

### Task and Prompt Formatting

Modern SFT relies heavily on **structured role-based templates**, such as ChatML or LLaMA-style formats.

**Example**
```text
<|system|> You are a helpful assistant. <|end_of_text|>
<|user|> Summarize this article. <|end_of_text|>
<|assistant|> The article discusses...
```

## Why Formatting Matters

- Separates intent, context, and response  
- Enables multi-turn dialogue modeling  
- Reduces ambiguity during inference  

Formatting does more than separate roles.

- **Implicit policy learning**  
  Role tokens and system messages act as soft constraints that the model internalizes as behavioral priors.

- **Gradient routing**  
  Loss masking combined with role tokens ensures gradients primarily shape response behavior rather than prompt reconstruction.

- **Inference controllability**  
  Well-designed templates allow downstream systems to inject safety, tools, or routing instructions without retraining.

- **Multi-turn state compression**  
  Structured formatting helps the model compress dialogue history into latent state representations instead of treating each turn independently.


---

## Prompt Diversity as Regularization

Prompt diversity is best understood as a **regularization strategy**, not just data augmentation.

### Semantic Diversity
Maintains coverage of distinct internal circuits:
- Symbolic reasoning and math
- Program synthesis and execution-style reasoning
- Creative and stylistic generation
- Factual recall under instruction pressure
- Conversational grounding

### Structural Diversity
Reduces shortcut learning:
- Paraphrased intents prevent lexical memorization
- Variable verbosity avoids length priors
- Explicit vs implicit constraints force instruction parsing

**Key insight**  
Insufficient structural diversity causes the model to learn *response templates* instead of *instruction semantics*.

---

## 5.1.3 Human vs Synthetic Supervision

### Human-Labeled Data

**Strengths**

- High factual accuracy  
- Strong alignment with human preferences  
- Better safety and tone control  

**Limitations**

- Expensive  
- Slow to scale  
- Limited coverage of edge cases  

---

### Synthetic Supervision

Modern post-training pipelines rely heavily on synthetic data generation.

#### Self-Instruct

- Start with a small human-curated seed set  
- Use a strong model to generate new instructions and responses  
- Filter for quality  

#### Evol-Instruct as Curriculum Learning

Evol-Instruct implicitly creates a **difficulty curriculum**:
- Base instruction
- Added constraints
- Multi-hop or multi-objective reasoning
- Strict formatting or safety requirements

This improves:
- Instruction decomposition
- Constraint satisfaction
- Planning depth

---

### Rejection Sampling (Best-of-N)

A widely used but often overlooked step in SFT.

**Process**

1. Generate $K$ responses per prompt  
2. Score them using:  
   - A reward model, or  
   - A stronger reference model  
3. Select the best response  
4. Fine-tune on the selected outputs  

**Key properties**

- Sharpens instruction adherence without policy gradients
- Reduces variance compared to RLHF
- Biases the model toward high-reward modes

**Advanced risk**

- Over-optimization toward the reward model
- Reduced output diversity
- Reward hacking if the scorer is weak


---

## 5.1.4 Data Quality over Quantity

### The LIMA Hypothesis

**“Less Is More for Alignment”**

**Key insight:**  

~1,000 extremely high-quality examples can outperform tens of thousands of noisy ones  

**Implications:**  

- Careful curation matters more than scale  
- Labeler expertise is critical  
- Reduces overfitting and style bias  

---

### Typical SFT Data Mix

A strong SFT dataset often includes:

- Reasoning and step-by-step explanations  
- Creative and stylistic writing  
- Coding and math problems  
- Safety and refusal examples  
- Multi-turn conversations  

---

## 5.1.5 Training Optimizations and Stability

| Technique       | Purpose         | Explanation                                                                             |
|-----------------|----------------|-----------------------------------------------------------------------------------------|
| Packing         | Throughput     | Concatenates multiple short samples into a single context window to avoid padding waste |
| Loss Masking    | Correct gradients | Computes loss only on assistant tokens                                                  |
| NEFTune         | Generalization | Adds noise to embeddings during SFT to prevent token-level overfitting                  |
| Low learning rate | Stability     | Typical values are $1e^{-6}$ to $5e^{-6}$                                               |
| Dropout         | Regularization | Reduces stylistic memorization                                                          |

### Packing vs Padding

In Supervised Fine-Tuning, training samples vary widely in length. How these samples are batched has a **direct impact on compute efficiency, gradient quality, and training stability**.

---

#### Padding

**What it is**  
All sequences in a batch are padded to the length of the longest sequence using `[PAD]` tokens.

**Example**

- `Seq 1: [x x x x x x]`
- `Seq 2: [x x x PAD PAD PAD]`
- `Seq 3: [x x x x PAD PAD]`


**Why it is inefficient**
- Attention, feedforward layers, and layer norms still execute on padded tokens
- Memory bandwidth and FLOPs are wasted on tokens that contribute no gradient
- Effective tokens per batch can drop sharply when length variance is high

**Impact at scale**
- Lowers tokens processed per second
- Increases training cost
- Reduces gradient signal density

---

#### Packing

**What it is**  
Multiple short samples are concatenated into a single long sequence up to the model’s maximum context length. Each sample is separated by an EOS or special boundary token, and loss masking prevents cross-sample leakage.

**Example**

`[Prompt₁ → Response₁ <EOS> Prompt₂ → Response₂ <EOS> Prompt₃ → Response₃]`



**Why it is efficient**
- Nearly every token contributes to loss
- Attention computation is fully utilized
- Higher effective batch token count without increasing memory

---

#### Why Packing Improves GPU Utilization

Packing increases:
- **Arithmetic intensity** by reducing idle FLOPs
- **Token density per batch**, improving gradient signal-to-noise ratio
- **Throughput**, often by 2x to 3x in instruction-tuning workloads where samples are short

This is especially impactful for:
- Instruction datasets with short prompts
- Chat-style SFT data
- Small to medium batch sizes constrained by memory

---

#### Important Implementation Details

- **Loss masking** must reset at each sample boundary
- **Attention masking** must prevent tokens from attending across examples
- **EOS handling** is critical to avoid information leakage
- **Position indices** may need resetting depending on the architecture

Incorrect packing can cause:
- Cross-example contamination
- Training instability
- Spurious memorization

---

#### Padding vs Packing Summary

| Aspect | Padding | Packing |
|------|--------|--------|
| Compute efficiency | Low | High |
| Token utilization | Sparse | Dense |
| Training speed | Slow | Fast |
| Implementation complexity | Simple | Moderate |
| Risk of leakage | None | Requires care |

---

#### Interview Insight

Packing is not just an optimization. It changes the **effective learning dynamics** by increasing gradient density and stabilizing updates, which is why it is now standard practice in large-scale SFT pipelines.

---

## 5.1.6 Overfitting and Catastrophic Forgetting

### Catastrophic Forgetting

The model loses general reasoning or factual knowledge learned during pre-training.

**Causes**

- Narrow SFT domain  
- High learning rates  
- Full fine-tuning on small datasets  

**Mitigations**

- Mix 5–10% pre-training style data  
- Use PEFT methods such as LoRA  
- Lower learning rates  
- Shorter training schedules  

---

### Overfitting

The model learns labeler-specific style rather than task intent.

**Symptoms**

- Over-politeness  
- Repetitive phrasing  
- Template-like answers  

**Mitigations**

- Prompt diversity  
- Early stopping  
- Noise injection such as NEFTune  

---

## 5.1.7 LoRA vs Full Fine-Tuning

### LoRA and PEFT

- Low compute cost  
- Preserves base model knowledge  
- Lower risk of catastrophic forgetting  
- Preferred for alignment and style shifts  

### Full Fine-Tuning

- Needed for large domain shifts  
- Higher risk of forgetting  
- Requires careful regularization  

**Rule of thumb:**  

- Behavior change → LoRA  
- Knowledge change → Full fine-tuning  

---

## 5.1.8 Common Failure Modes After SFT

### Increased Hallucinations

Often caused by **knowledge contradiction**.

If SFT data conflicts with pre-training facts, the model may prioritize format compliance over correctness.

**Mitigations**

- Fact-consistent SFT data  
- Retrieval-augmented generation  
- Post-SFT preference optimization  

---

## 5.1.9 SFT vs Pre-training Summary

| Aspect         | Pre-training     | Supervised Fine-Tuning |
|----------------|-----------------|-----------------------|
| Objective      | World modeling  | Behavior alignment    |
| Data scale     | Trillions of tokens | 10k to 100k samples |
| Loss           | Full sequence NTP | Masked response NTP |
| Compute        | Massive         | Moderate             |
| Primary risk   | Under-training  | Overfitting and forgetting |

---

## 5.1.10 Interview-Level Takeaways

- SFT aligns behavior rather than knowledge  
- Loss masking is essential  
- Data quality dominates data scale  
- Synthetic data is now standard  
- Rejection sampling is widely used  
- Forgetting is a first-order concern  
- SFT sets the foundation for RLHF and preference optimization  
