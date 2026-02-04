# 🧠 LLM-Finetuning: Parameter-Efficient Adaptation of Large Language Models

This repository explores **Parameter-Efficient Fine-Tuning (PEFT)** methods for adapting large language models (LLMs) such as **Llama**, **Mistral**, and **Gemma** to downstream tasks using minimal computational resources.

> 🔗 **Documentation:** [View Detailed Docs](https://saurabhiit2007.github.io/LLM-Training-Hub/)

---

## 📘 Overview

Traditional fine-tuning of large models is computationally expensive and memory-intensive.  
This project investigates **PEFT methods** — such as **LoRA** and **QLoRA** — which modify only a small subset of model parameters while keeping the base model frozen.

---

## 🧩 Datasets

We experiment with one dataset each for **three task domains**:

### 1. 🧮 Classification Task
Used for evaluating models’ ability to adapt to discrete label prediction.

### 2. ✍️ Next Token Prediction (Causal LM)
**Dataset:** `wikitext-2-raw-v1`

The **WikiText** dataset contains raw text from verified Wikipedia articles.

#### Example
```python
{"text": "In 2006, the United States launched the Pluto mission to explore the dwarf planet."}
```

#### Dataset Statistics
| Split | Size (examples) | Tokens (approx.) |
|--------|------------------|------------------|
| Train | 37,218 | ~2.1M |
| Validation | 3,760 | ~0.2M |
| Test | 4,358 | ~0.2M |

#### Task Objective
Causal language modeling (next-token prediction):

| Input | Target |
|--------|---------|
| A sequence of tokens | The next token in the sequence |

**Preprocessing:**
- **Tokenization:** Convert text to token IDs.  
- **Concatenation & Grouping:** Merge sentences and split into fixed-size blocks (e.g., 128 tokens).  
- **Labels:** Shifted by one position (teacher forcing).

Example:
```
Input IDs:  [101, 234, 567, 890, 123, 456, 789, 321, 654, 987]
Labels:     [234, 567, 890, 123, 456, 789, 321, 654, 987, 102]
```

### 3. 💬 Instruction Following
Used to test instruction-tuned behavior and generalization.

---

## 🏗️ Models Used

| Model Name | Source |
|-------------|---------|
| **Llama-2-7B** | [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) |
| **Mistral-7B-v0.1** | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) |

---

## 🧠 Methods Explored

### 1. LoRA (Low-Rank Adaptation of LLMs)
LoRA adds low-rank matrices to attention and projection layers, allowing efficient fine-tuning of large models.

**Advantages:**
- Minimal additional parameters.
- Fast training.
- Compatible with full model architectures.

📘 **[LoRA Documentation →](https://saurabhiit2007.github.io/LLM-Finetuning/)**

#### 🔧 Hyperparameters
- **Rank (r):** {4, 8, 16, 128}  
- **Alpha (α):** {16, 32, 256}  
- **Target Modules:** `q_proj`, `v_proj`  
- **Learning Rate:** 2e-4  
- **Block Size:** 128 tokens  
- **Batch Size:** 4 × 8 (gradient accumulation)

---

## 📊 Results

### 1. 🧮 Classification Task
### **Dataset: xxxxxxx**

| Model | Training Type | LoRA Config (r, α) | Time/Epoch (min) | Accuracy (%) ↑ | Memory (GB) |
|--------|----------------|--------------------|------------------|----------------|--------------|
| **Model_1** | Full | - | 130 | 82.0 | 16 |
|  | LoRA | (4, 16) | 50 | 81.5 | 6 |
|  | LoRA | (8, 32) | 55 | 81.8 | 7 |
| **Model_2** | Full | - | 210 | 87.0 | 24 |
|  | LoRA | (4, 16) | 80 | 86.5 | 9 |
|  | LoRA | (8, 32) | 85 | 86.8 | 10 |
---

### 2. ✍️ Next Token Prediction (Causal LM)
### **Dataset: wikitext-2-raw-v1**

| Model | Training Type | LoRA Config (r, α) | Time/Epoch (min) | Perplexity ↓ | Memory (GB) |
|--------|----------------|--------------------|------------------|--------------|--------------|
| **Mistral-7B-v0.1** | Full | - | 120 | 85.2 | 16 |
|  | LoRA | (8, 16) | 45 | 84.8 | 6 |
|  | LoRA | (16, 32) | 50 | 85.0 | 7 |
|  | LoRA | (128, 256) | 50 | 85.0 | 7 |
| **Llama-2-7B** | Full | - | 200 | 88.0 | 24 |
|  | LoRA | (8, 16) | 75 | 87.5 | 9 |
|  | LoRA | (16, 32) | 80 | 87.8 | 10 |
|  | LoRA | (128, 256) | 80 | 87.8 | 10 |
---

### 3. 💬 Instruction Following
### **Dataset: xxxxxxxxxx**

| Model | Training Type | LoRA Config (r, α) | Time/Epoch (min) | Perplexity ↓ | GPU/CPU Memory (GB) |
|--------|----------------|--------------------|------------------|--------------|--------------|
| **Mistral-7B-v0.1** | Full | - | 120 | 85.2 | 16 |
|  | LoRA | (8, 16) | 45 | 84.8 | 6 |
|  | LoRA | (16, 32) | 50 | 85.0 | 7 |
|  | LoRA | (128, 256) | 50 | 85.0 | 7 |
| **Llama-2-7B** | Full | - | 200 | 88.0 | 24 |
|  | LoRA | (8, 16) | 75 | 87.5 | 9 |
|  | LoRA | (16, 32) | 80 | 87.8 | 10 |
|  | LoRA | (128, 256) | 80 | 87.8 | 10 |
---

### 2. QLoRA *(In Progress)*
QLoRA applies **4-bit quantization** with **LoRA adapters** to reduce memory usage while maintaining performance parity with full precision fine-tuning.

---

## ⚙️ Environment Setup

```bash
# Clone repository
git clone https://github.com/saurabhiit2007/LLM-Finetuning.git
cd LLM-Finetuning

# Create environment (using uv or venv)
uv init . --python 3.12
uv add torch transformers peft datasets accelerate bitsandbytes

# Run LoRA training
python train_lora.py --model meta-llama/Llama-3-8b --dataset wikitext-2-raw-v1
```

---

## 🧾 Citation
If you use this work or build upon it, please consider citing:

```
@misc{goyal2025lora,
  author       = {Saurabh Goyal},
  title        = {LLM-Finetuning: Parameter-Efficient Fine-Tuning of Large Language Models},
  year         = {2025},
  url          = {https://github.com/saurabhiit2007/LLM-Finetuning}
}
```

---

## 🧩 To-Do / Future Work
- [ ] Complete QLoRA experiments  
- [ ] Add instruction-following dataset results  
- [ ] Include comparative plots for speed vs. accuracy trade-offs  
- [ ] Integrate evaluation metrics beyond perplexity (BLEU, F1)  

---

## 🧑‍💻 Author
**Saurabh Goyal**  
[GitHub](https://github.com/saurabhiit2007) • [LinkedIn](https://www.linkedin.com/in/saurabh-goyal/)

---

🪶 *“Small changes to parameters, big changes to capability.”*
