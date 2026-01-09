## 📦 Decoding Strategies

### 1. Overview
Large Language Models output a probability distribution over the vocabulary at each decoding step. A **decoding strategy** defines how the next token is selected from this distribution.

This page covers five commonly used decoding strategies:

1. Greedy decoding  
2. Beam search  
3. Temperature sampling  
4. Top-k sampling  
5. Top-p sampling (nucleus sampling)

---

### 2. Decoding Strategies Explained with examples

Toy probability distribution used in examples. Assume the model predicts the next token after:

    **"The cat sat on the"**

| Token | Probability |
|------|-------------|
| mat | 0.40 |
| floor | 0.25 |
| sofa | 0.15 |
| bed | 0.10 |
| roof | 0.05 |
| moon | 0.03 |
| pizza | 0.02 |

---

### 2.1. Greedy Decoding

#### Idea
Always select the token with the highest probability.

#### Algorithm
    next_token = argmax(probabilities)
    Highest probability token is `mat`.
    Output: The cat sat on the mat

#### Edge Case
    If probabilities are very close: A: 0.31, B: 0.30, C: 0.29
    Greedy decoding always selects `A`, even when the model is uncertain.  
    
    This often leads to repetitive or dull outputs.

#### When to use
- Debugging
- Baselines
- Deterministic generation

#### Python example
```python
import torch

probs = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
next_token = torch.argmax(probs)
print(next_token.item())
```

---

### 2.2 Beam Search

#### Idea
Beam search keeps multiple candidate sequences at each decoding step instead of a single one. It selects the sequence with the highest **overall probability**, not just the best local choice.

#### Algorithm
1. Maintain **B beams**, where B is the beam width  
2. At each step, expand every beam with all possible next tokens  
3. Compute cumulative log probability for each expanded sequence  
4. Keep the top B sequences  
5. Repeat until an end condition is met

#### Example
> Assume the next-token probabilities after:
> 
> **"The cat sat on the"**
>
>| Token | Probability |
>|------|-------------|
>| mat | 0.40 |
>| floor | 0.25 |
>| sofa | 0.15 |
>
>**Beam width = 2**
>
>**Step 1**
> 
>- Beam 1: `"mat"` score = log(0.40)
>- Beam 2: `"floor"` score = log(0.25)
>
>**Step 2**
> 
>- `"mat → quietly"` score = log(0.40) + log(0.30)
>- `"floor → loudly"` score = log(0.25) + log(0.50)
>
>Even if `"quietly"` was locally better, `"floor → loudly"` may win due to higher cumulative probability.
>
>Final output is the sequence with the highest total score.

---

#### Edge Case

Beam search tends to favor **safe, high-probability continuations**, which can reduce diversity. This behavior becomes obvious in conversational or creative tasks.

>Assume the model is generating the next phrase after:
>
>**"I think that"**
>
>At a certain step, the model assigns probabilities like:
>
>| Token | Probability |
>|------|-------------|
>| the | 0.35 |
>| we | 0.30 |
>| this | 0.15 |
>| pizza | 0.10 |
>| unicorn | 0.10 |
>
>With **beam width = 3**:
>
>All beams will keep continuations starting with:
> `"the"`
> `"we"`
> `"this"`
>
>Tokens like `"pizza"` and `"unicorn"` are discarded early because their probabilities are lower.
>
>As decoding continues, beams converge to similar phrases:
> 
>- I think that the best way to...
>- I think that we should...
>- I think that this is...
>
>All beams are grammatically correct but **nearly identical**.
>
>If **top-p sampling** is used instead:
> 
>- Tokens like `"pizza"` or `"unicorn"` may occasionally be sampled
>- Outputs become more diverse:
>
>       - I think that pizza could solve this
>       - I think that unicorn stories are fun

#### When to use beam search
- Machine translation  
- Speech recognition  
- Structured text generation  
- Tasks where correctness matters more than diversity

#### When not to use beam search
- Chatbots  
- Story generation  
- Creative writing  
- Conversational agents

#### Python example (simplified)

```python
from heapq import nlargest
import math

def beam_search_step(beams, probs, beam_width):
    new_beams = []
    for seq, score in beams:
        for i, p in enumerate(probs):
            new_seq = seq + [i]
            new_score = score + math.log(p)
            new_beams.append((new_seq, new_score))
    return nlargest(beam_width, new_beams, key=lambda x: x[1])

# Initial beam
beams = [([], 0.0)]
probs = [0.40, 0.25, 0.15]

beams = beam_search_step(beams, probs, beam_width=2)
print(beams)
```

---

### 2.3 Temperature Sampling

#### Idea
Temperature controls how random the next-token selection is by scaling the model logits before applying softmax.

It does not change which tokens are possible.  
It changes **how strongly the model prefers high-probability tokens**.

#### Formula
$$p_i = \text{softmax}(\text{logits}_i / T)$$

Where:

- `T` is the temperature
- lower `T` sharpens the distribution
- higher `T` flattens the distribution

#### Effect of temperature

| Temperature | Behavior |
|------------|---------|
| T < 1 | More deterministic |
| T = 1 | Original distribution |
| T > 1 | More random |

#### Example
>Assume the next-token probabilities are:
>
>| Token | Probability |
>|------|-------------|
>| mat | 0.40 |
>| floor | 0.25 |
>| sofa | 0.15 |
>| bed | 0.10 |
>| roof | 0.05 |
>| moon | 0.03 |
>| pizza | 0.02 |
>
>**Low temperature (T = 0.3)**
>
>- Distribution becomes very sharp
>- `mat` dominates even more
>
>Output:
>The cat sat on the mat
>
>This behaves almost like greedy decoding.

>**High temperature (T = 1.5)**
>
>- Distribution becomes flatter
>- Low-probability tokens become more likely
>
>Possible output:
>The cat sat on the moon

#### Edge Case

With very high temperature:

| Token | Probability |
|------|-------------|
| mat | 0.18 |
| floor | 0.17 |
| sofa | 0.16 |
| bed | 0.15 |
| roof | 0.14 |
| moon | 0.10 |
| pizza | 0.10 |

The model loses strong preferences and may generate incoherent text:
    
    The cat sat on pizza quantum sky

#### When temperature helps
- Creative writing
- Brainstorming
- Open-ended dialogue

#### When temperature hurts
- Factual tasks
- Code generation
- Structured outputs

#### Python example

```python
import torch

logits = torch.log(torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]))
temperature = 1.2

scaled_logits = logits / temperature
probs = torch.softmax(scaled_logits, dim=0)

next_token = torch.multinomial(probs, 1)
print(next_token.item())
```

> Note: Temperature controls randomness, not feasibility.
It is usually combined with top-p or top-k sampling to avoid incoherent outputs.
 

### 2.4 Top-k Sampling

#### Idea
Top-k sampling restricts the model to sample only from the **K most probable tokens** at each decoding step. This prevents extremely unlikely tokens from being selected while still allowing randomness.

#### Algorithm
1. Sort all tokens by probability  
2. Keep only the top K tokens  
3. Renormalize their probabilities  
4. Sample one token  

### Example
>Assume the next-token probabilities are:
>
>| Token | Probability |
>|------|-------------|
>| mat | 0.40 |
>| floor | 0.25 |
>| sofa | 0.15 |
>| bed | 0.10 |
>| roof | 0.05 |
>| moon | 0.03 |
>| pizza | 0.02 |
>
>**Top-k with k = 3**
>
>Tokens kept:
> 
>- mat
>- floor
>- sofa
>
>Tokens removed:
> 
>- bed, roof, moon, pizza
>
>Possible output: The cat sat on the sofa

#### Edge Case

**Flat probability distribution**

Assume: A: 0.11, B: 0.10, C: 0.10, D: 0.10, E: 0.10, F: 0.10, G: 0.10

With `k = 3`:

- Only A, B, C are considered
- D, E, F, G are removed despite being equally likely

This makes top-k **sensitive to the choice of K** and blind to the shape of the distribution.

#### When top-k works well
- Moderate creativity with controlled randomness
- General text generation
- Chat systems with fixed diversity constraints

#### When top-k works poorly
- Highly uncertain distributions
- Long-form creative writing
- Prompts with many equally valid continuations

#### Python example

```python
import torch

def top_k_sampling(probs, k):
    topk_probs, topk_idx = torch.topk(probs, k)
    topk_probs = topk_probs / topk_probs.sum()
    sampled = torch.multinomial(topk_probs, 1)
    return topk_idx[sampled]

probs = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
token = top_k_sampling(probs, k=3)
print(token.item())
```

> Note: Top-k sampling fixes the number of candidate tokens regardless of model confidence.
This makes it simpler than top-p but less adaptive in practice.


### 2.5 Top-p Sampling (Nucleus Sampling)

#### Idea
Top-p sampling selects the **smallest possible set of tokens** whose cumulative probability is at least `p`,  
then samples from that set. Unlike top-k, the number of candidate tokens **changes dynamically** based on model confidence.

#### Algorithm
1. Sort tokens by probability in descending order  
2. Add tokens until cumulative probability ≥ `p`  
3. Renormalize probabilities within this set  
4. Sample one token  

#### Example

>Assume the next-token probabilities are:
>
>| Token | Probability |
>|------|-------------|
>| mat | 0.40 |
>| floor | 0.25 |
>| sofa | 0.15 |
>| bed | 0.10 |
>| roof | 0.05 |
>| moon | 0.03 |
>| pizza | 0.02 |
>
>**Top-p with p = 0.9**
>
>Cumulative probability:
> 
>- mat → 0.40  
>- floor → 0.65  
>- sofa → 0.80  
>- bed → 0.90  
>
>Tokens selected:
> 
>- mat
>- floor
>- sofa
>- bed
>
>Possible output: The cat sat on the bed

#### Edge Case (Key Difference from Top-k)

>**Highly confident model**
>
>Assume: A: 0.85, B: 0.07, C: 0.03, D: 0.03, E: 0.02
>
>With `p = 0.9`:
> 
>- Selected tokens: A, B  
>- Effective K = 2
>
>With top-k (k = 5):
> 
>- Selected tokens: A, B, C, D, E  
>
>Top-p automatically reduces randomness when the model is confident.

#### Another Edge Case

>**Uncertain model**
>
>Assume: A: 0.20, B: 0.20, C: 0.20, D: 0.20, E: 0.20
>
>With `p = 0.9`:
> 
>- Selected tokens: A, B, C, D, E  
>- Effective K = 5
>
>Top-p expands the candidate set when uncertainty is high.

#### When top-p works well
- Conversational agents
- Long-form text generation
- Creative writing with coherence

#### When top-p works poorly
- Strictly deterministic tasks
- Code generation with exact formatting requirements

#### Python example

```python
import torch

def top_p_sampling(probs, p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    cutoff_mask = cumulative <= p
    cutoff_mask[cutoff_mask.sum()] = True

    filtered_probs = sorted_probs[cutoff_mask]
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled = torch.multinomial(filtered_probs, 1)
    return sorted_idx[cutoff_mask][sampled]

probs = torch.tensor([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
token = top_p_sampling(probs, p=0.9)
print(token.item())
```

> Note : Top-p sampling adapts to the probability distribution shape,
making it more robust than top-k for real-world language generation.


### 3. Pros and Cons of Decoding Strategies in Large Language Models

#### 3.1 Greedy Decoding

#### Pros
- Extremely fast and simple
- Fully deterministic and reproducible
- Easy to debug and analyze
- Works well when the model is very confident

#### Cons
- No diversity at all
- Easily gets stuck in repetitive loops
- Early mistakes cannot be corrected
- Often produces dull or incomplete responses
- Poor performance for long or open-ended generation

---

#### 3.2 Beam Search

#### Pros
- Optimizes global sequence likelihood
- Reduces early local decision errors
- Produces fluent and grammatically correct text
- Effective for tasks with a single correct output

#### Cons
- Computationally expensive
- Produces generic and safe outputs
- Very low diversity
- All beams often converge to similar sequences
- Performs poorly for dialogue and creative tasks

---

#### 3.3 Temperature Sampling

#### Pros
- Simple and intuitive control over randomness
- Enables creative and diverse outputs
- Easy to combine with other sampling methods
- Useful for brainstorming and storytelling

#### Cons
- High temperature can cause incoherent text
- Low temperature collapses to greedy behavior
- Does not prevent sampling of very unlikely tokens
- Sensitive to temperature tuning

---

#### 3.4 Top-k Sampling

#### Pros
- Prevents extremely low-probability tokens
- Provides controlled randomness
- Simple to implement
- More diverse than greedy and beam search

#### Cons
- Fixed K ignores distribution shape
- Sensitive to the choice of K
- Removes valid tokens in flat distributions
- Not adaptive to model confidence

---

#### 3.5 Top-p Sampling (Nucleus Sampling)

#### Pros
- Adapts automatically to model confidence
- Better diversity-quality tradeoff than top-k
- Stable across different prompts
- Widely used in modern chat models

#### Cons
- Slightly more complex than top-k
- Still stochastic and non-deterministic
- Can include many tokens in very flat distributions
- Less suitable for strictly deterministic tasks

---

### 4. High-level Comparison

| Strategy | Diversity | Determinism | Adaptivity | Typical Usage |
|--------|-----------|-------------|------------|---------------|
| Greedy | Very low | High | No | Baselines, debugging |
| Beam Search | Low | Medium | No | Translation, ASR |
| Temperature | Medium to high | Low | Partial | Creative text |
| Top-k | Medium | Low | No | General generation |
| Top-p | Medium to high | Low | Yes | Chat and dialogue |

---



 