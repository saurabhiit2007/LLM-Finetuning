### 📦1. Speculative Decoding: Overview

Speculative decoding reduces **inference latency** in decoder-only LLMs while preserving the **exact output distribution** of a large target model.

#### 1.1 Why standard decoding is slow

In standard autoregressive decoding:

- The target model generates **one token per forward pass**
- The model is large and expensive
- Latency grows linearly with output length

KV cache reduces computation but does not remove the sequential bottleneck.

#### 1.2 Core idea

Speculative decoding separates **token proposal** from **token verification**:

- A **draft model** proposes multiple tokens cheaply
- A **target model** verifies them efficiently

If most draft tokens are accepted, multiple tokens are generated per expensive target model forward pass.

---

### 📦2. Background: How Decoding Works in Decoder-Only LLMs

Before speculative decoding, it is critical to understand standard autoregressive decoding.

#### 2.1 Autoregressive modeling assumption

A decoder-only LLM models a sequence of tokens using the following factorization:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})
$$

Key points:

- Tokens are generated **left to right**
- Each new token depends on **all previous tokens**
- There is no notion of predicting multiple future tokens independently

#### 2.2 What happens in one forward pass

Assume the current sequence length is $T$.

#### Step 1: Embedding

Each token $x_t$ is mapped to a vector representation:

$$
\mathbf{e}_t = \text{TokenEmbed}(x_t) + \text{PosEmbed}(t)
$$

#### Step 2: Masked self-attention

For each token position $t$:

$$
\mathbf{q}_t = \mathbf{e}_t W_Q,\quad
\mathbf{k}_t = \mathbf{e}_t W_K,\quad
\mathbf{v}_t = \mathbf{e}_t W_V
$$

Attention scores are computed as:

$$
\alpha_{t,i} = \frac{\mathbf{q}_t \cdot \mathbf{k}_i}{\sqrt{d_k}}
$$

A **causal mask** ensures token $t$ can only attend to tokens $i \le t$.

The attended representation is:

$$
\mathbf{a}_t = \sum_{i=1}^{t} \text{softmax}(\alpha_{t,i}) \mathbf{v}_i
$$

This is followed by a linear projection:

$$
\mathbf{o}_t = \mathbf{a}_t W_O
$$

#### Step 3: Feed Forward Network (FFN)

Each token is processed independently:

$$
\mathbf{h}_t = \text{FFN}(\mathbf{o}_t)
$$

Residual connections and layer normalization are applied around both attention and FFN blocks.

This completes one decoder layer. The same process repeats across multiple stacked layers.

#### 2.3 Computing logits and selecting the next token

After the final decoder layer, each token position has a hidden state $\mathbf{h}_t$.

These are projected to vocabulary logits:

$$
\mathbf{z}_t = \mathbf{h}_t W_{\text{vocab}}
\quad \text{where } \mathbf{z}_t \in \mathbb{R}^{|V|}
$$

Important clarification:

- Logits are computed **for every token position**
- Softmax is applied **over the vocabulary**
- During decoding, **only the last position is used**

$$
P(x_{T+1} \mid x_{\le T}) = \text{softmax}(\mathbf{z}_T)
$$

A token is selected using greedy decoding or sampling and appended to the sequence.

#### 2.4 Autoregressive decoding loop

For each generated token:

1. Run the Transformer forward pass
2. Take logits from the last token position
3. Apply softmax over the vocabulary
4. Select one token
5. Append it to the sequence
6. Repeat

This process continues until an end-of-sequence token is produced or a maximum length is reached.

#### 2.5 Key limitation of standard decoding

- Each generated token requires a new forward pass of the model
- Latency grows linearly with output l


### 📦3. Step-by-Step Algorithm for Speculative Decoding

This section describes the speculative decoding algorithm precisely, step by step, focusing on what each model does and why it is needed.

Assume:

- Prompt tokens: $x$
- Draft model: $q$
- Target model: $p$
- Draft length: $k$
- Drafted tokens: $y_1, y_2, \dots, y_k$

#### Step 1: Draft model proposes tokens

The draft model generates tokens autoregressively, starting from the prompt:

$$
y_i \sim q(\cdot \mid x, y_{<i}) \quad \text{for } i = 1 \dots k
$$

Key points:

- This step is fast because the draft model is small
- Tokens are sampled, not greedily selected
- The draft model also records the probability of each sampled token

#### Step 2: Target model verifies the draft

The target model is run **once** on the combined sequence: $[x, y1, y2, ..., yk]$

This produces target model probabilities:

$$
p(y_i \mid x, y_{<i}) \quad \text{for } i = 1 \dots k
$$

Important clarification:

- The target model naturally computes logits for all positions
- Only logits corresponding to the drafted tokens are used
- Logits for the prompt tokens are ignored

#### Step 3: Acceptance test

Each drafted token is accepted independently using:

$$
\alpha_i = \min\left(1, \frac{p(y_i \mid x, y_{<i})}{q(y_i \mid x, y_{<i})}\right)
$$

Procedure:

1. Sample $u \sim \text{Uniform}(0, 1)$
2. Accept token $y_i$ if $u \le \alpha_i$
3. Stop at the first rejected token

#### Step 4: Rejection handling and fallback

If token $y_j$ is rejected:

- Tokens $y_j, y_{j+1}, \dots, y_k$ are discarded
- The next token is sampled directly from the target model:
  $$
  x_{\text{next}} \sim p(\cdot \mid x, y_{<j})
  $$
- Speculative decoding restarts from the new prefix

If no token is rejected, all $k$ draft tokens are accepted.

#### Why this preserves correctness

- The acceptance rule implements rejection sampling
- Bias introduced by the draft model is corrected
- The final output distribution exactly matches the target model

This guarantees that speculative decoding is statistically equivalent to standard decoding with the target model.


### 📦4. Why Speculative Decoding Is Faster

Speculative decoding reduces inference latency by decreasing how often the expensive target model must be executed.

#### 4.1 Standard decoding vs speculative decoding

**Standard decoding**

- One target model forward pass per generated token
- For $N$ tokens, $N$ forward passes are required

**Speculative decoding**

- The draft model proposes $k$ tokens cheaply
- A single target model forward pass verifies up to $k$ tokens
- Multiple tokens can be generated per target model invocation

#### 4.2 Source of the speedup

The speedup comes from two properties:

- The draft model is significantly cheaper than the target model
- The target model can evaluate multiple draft tokens in parallel

If the acceptance rate is high, the target model is called much less frequently.

#### 4.3 What speculative decoding does not optimize

Speculative decoding does not reduce:

- The total number of FLOPs in the target model
- The per-token computation inside the Transformer

It primarily reduces **latency**, not theoretical compute.

#### 4.4 Practical speedups

In practice, speculative decoding often achieves:

- 1.5x to 3x latency improvement
- Higher gains when draft and target distributions are close

Actual speedup depends on model sizes, hardware, and acceptance rate.

### 5. Relationship to Logits for All Tokens

Speculative decoding does not introduce a new requirement to compute logits for all tokens.

#### 5.1 Standard Transformer behavior

A Transformer forward pass naturally produces:

- One hidden state per token position
- One vocabulary logits vector per token position

This is true during both training and inference.

#### 5.2 How logits are used in speculative decoding

During verification:

- The target model is run once on the prompt plus draft tokens
- Logits for prompt tokens are ignored
- Only logits corresponding to the draft token positions are used

Speculative decoding simply reuses standard per-position logits.

#### 5.3 What speculative decoding does not do

Speculative decoding does not:

- Perform softmax over token positions
- Predict future tokens independently
- Generate tokens in parallel from the target model

The target model still defines an autoregressive distribution.

### 6. Interaction with KV Cache

KV cache improves performance in speculative decoding but does not change the algorithm.

#### 6.1 KV cache in the draft model

- The draft model maintains its own KV cache
- Draft tokens are generated autoregressively
- KV cache allows fast token proposal

#### 6.2 KV cache in the target model

- The target model computes KV cache for the entire speculative window
- KV states corresponding to accepted tokens are reused
- KV states for rejected tokens are discarded

#### 6.3 Why KV cache matters

KV cache:

- Avoids recomputing attention for previously processed tokens
- Reduces per-step computation
- Improves practical throughput

KV cache affects efficiency only, not correctness.

---
