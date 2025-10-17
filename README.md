# LLM-Finetuning

Using parameter-efficient fine-tuning (PEFT) to fine-tune the LLM model.

Datasets:

The training and analysis was done using 2 datasets from 3 task domains each:
1. Classification task

2. Next token Prediction
```
Wikitext : The WikiText datasets are extracted from Wikipedia articles, so it’s basically raw text with paragraphs and sentences, sometimes with headings.
Each example is a dictionary like:
{'text': "In 2006, the United States launched the Pluto mission to explore the dwarf planet."}

Specifics of wikitext-2-raw-v1

Example:
    "Pierre-Auguste Renoir was a French artist who was a leading painter in the development of the Impressionist style."
    "He was born on February 25, 1841, in Limoges, France."
    "Renoir's works include 'Dance at Le Moulin de la Galette' and many portraits of women and children."

    Train Size: 37,218 (~2.1M tokens)
    Validation Size: 3,760 (~0.2M tokens)
    Test Size: 4,358 (~0.2M tokens)

    The task is causal language modeling (next-token prediction):
        Input: A sequence of tokens (words or subwords).
        Output / Target: The next token in the sequence.

    How the dataset is preprocessed for Causal LM

    - Tokenization: Convert each line to token IDs.
    - Grouping: Concatenate lines and split into blocks of fixed length (block_size, e.g., 128 tokens).
    - Labels: Same as input sequence (teacher forcing).
    
    For example, for a block of length 10 tokens:
    - Input IDs:  [101, 234, 567, 890, 123, 456, 789, 321, 654, 987]
    - Labels:     [234, 567, 890, 123, 456, 789, 321, 654, 987, 102]  # shifted by 1
```

3. Instruction following


Models Used
- [meta-llama/Llama-3-8b](https://huggingface.co/meta-llama/Llama-2-7b)
- Mistral-7B-v0.1
- mistralai/Mixtral-8x7B-v0.1
- google/gemma-7b


Methods Tried:

1. LORA: [![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://saurabhiit2007.github.io/LLM-Finetuning/)

Results:
## LoRA Finetuning Results

### Dataset 1: Dataset_A

| Model & Training              | LoRA Config (r, α) | Avg. Training Time/Epoch (min) | Performance (Perplexity)   | GPU/CPU Memory (GB) |
|-------------------------------|-------------------|---------------------------------|--------------|---------------------|
| **mistralai/Mistral-7B-v0.1** |                   |                                 |              |                     |
| Full                          | -                 | 120                             | 85.2         | 16                  |
| LoRA                          | r=8, α=16         | 45                              | 84.8         | 6                   |
| LoRA                          | r=16, α=32        | 50                              | 85.0         | 7                   |
| LoRA                          | r=128, α=256      | 50                              | 85.0         | 7                   |
|-----------------|-------------------|----------------|--------------|-------------|
| **Model_2**     |                   |                |              |             |
| Full            | -                 | 200            | 88.0         | 24          |
| LoRA            | r=4, α=16         | 75             | 87.5         | 9           |
| LoRA            | r=8, α=32         | 80             | 87.8         | 10          |
|-----------------|-------------------|----------------|--------------|-------------|
| **Model_3**     |                   |                |              |             |
| Full            | -                 | 150            | 83.5         | 20          |
| LoRA            | r=4, α=16         | 60             | 83.0         | 8           |
| LoRA            | r=8, α=32         | 65             | 83.2         | 9           |

---

### Dataset 2: Dataset_B

| Model & Training | LoRA Config (r, α) | Time/Epoch (min) | Accuracy (%) | Memory (GB) |
|-----------------|-------------------|----------------|--------------|-------------|
| **Model_1**     |                   |                |              |             |
| Full            | -                 | 130            | 82.0         | 16          |
| LoRA            | r=4, α=16         | 50             | 81.5         | 6           |
| LoRA            | r=8, α=32         | 55             | 81.8         | 7           |
|-----------------|-------------------|----------------|--------------|-------------|
| **Model_2**     |                   |                |              |             |
| Full            | -                 | 210            | 87.0         | 24          |
| LoRA            | r=4, α=16         | 80             | 86.5         | 9           |
| LoRA            | r=8, α=32         | 85             | 86.8         | 10          |
|-----------------|-------------------|----------------|--------------|-------------|
| **Model_3**     |                   |                |              |             |
| Full            | -                 | 160            | 84.0         | 20          |
| LoRA            | r=4, α=16         | 65             | 83.5         | 8           |
| LoRA            | r=8, α=32         | 70             | 83.7         | 9           |


2. QLoRA:
