import yaml
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from utils.utils import MetricsLoggerCallback, log_experiment


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config):

    # ---------------------------------------------------- Load Config ---------------------------------------------------- #

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    gen_cfg = config["generation"]

    # ------------------------------------------Load a pre-trained model-----------------------------------------------#

    model_name = model_cfg["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model Loading Done!!")

    # ------------------------------Define config with lora specific hyper-parameters----------------------------------#

    lora_config = LoraConfig(
        r=lora_cfg["r"],  # Rank of the low-rank matrices
        lora_alpha=lora_cfg["lora_alpha"],  # Scaling factor
        lora_dropout=lora_cfg["lora_dropout"],  # Dropout for regularization
        target_modules=lora_cfg["target_modules"],  # Target modules to apply LoRA
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )

    # Get the new model with lora adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Lora Adaptor Done!!")

    #------------------------------------------------Load dataset-----------------------------------------------------#

    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_config"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)

    dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Group texts into blocks of fixed length for causal LM training
    block_size = data_cfg["block_size"]

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = dataset.map(group_texts, batched=True, num_proc=4)

    if data_cfg["train_subset_size"]:
        train_dataset = lm_datasets["train"].select(range(data_cfg["train_subset_size"]))
    else:
        train_dataset = lm_datasets["train"]

    if data_cfg["eval_subset_size"]:
        eval_dataset = lm_datasets["validation"].select(range(data_cfg["eval_subset_size"]))
    else:
        eval_dataset = lm_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Dataset Prepared!!")
    # --------------------------------------------Train the model------------------------------------------------------#

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        fp16=train_cfg["fp16"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["evaluation_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
    )

    metrics_logger = MetricsLoggerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[metrics_logger]
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    print("Training Done!!")
    # --------------------------------------Evaluate on Validation dataset---------------------------------------------#

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    print(f"Perplexity: {perplexity}")
    print("Evaluation Done!!")

    # ------------------------------------------Test on new sample-----------------------------------------------------#

    inputs = tokenizer(gen_cfg["test_prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"]
        )
    print("Generated Text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Testing Done!!")

    # ------------------------------------------Log experiment-----------------------------------------------------#
    train_metrics = {
        "epoch_times": metrics_logger.epoch_times,
        "epoch_losses": metrics_logger.epoch_losses
    }
    log_experiment(config, eval_results, train_metrics, output_dir=train_cfg["results_logging"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file for hyper-parameters")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
import yaml
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from utils.utils import MetricsLoggerCallback, log_experiment


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config):

    # ---------------------------------------------------- Load Config ---------------------------------------------------- #

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    gen_cfg = config["generation"]

    # ------------------------------------------Load a pre-trained model-----------------------------------------------#

    model_name = model_cfg["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model Loading Done!!")

    # ------------------------------Define config with lora specific hyper-parameters----------------------------------#

    lora_config = LoraConfig(
        r=lora_cfg["r"],  # Rank of the low-rank matrices
        lora_alpha=lora_cfg["lora_alpha"],  # Scaling factor
        lora_dropout=lora_cfg["lora_dropout"],  # Dropout for regularization
        target_modules=lora_cfg["target_modules"],  # Target modules to apply LoRA
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )

    # Get the new model with lora adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Lora Adaptor Done!!")

    #------------------------------------------------Load dataset-----------------------------------------------------#

    dataset = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_config"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)

    dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Group texts into blocks of fixed length for causal LM training
    block_size = data_cfg["block_size"]

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = dataset.map(group_texts, batched=True, num_proc=4)

    if data_cfg["train_subset_size"]:
        train_dataset = lm_datasets["train"].select(range(data_cfg["train_subset_size"]))
    else:
        train_dataset = lm_datasets["train"]

    if data_cfg["eval_subset_size"]:
        eval_dataset = lm_datasets["validation"].select(range(data_cfg["eval_subset_size"]))
    else:
        eval_dataset = lm_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Dataset Prepared!!")
    # --------------------------------------------Train the model------------------------------------------------------#

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        fp16=train_cfg["fp16"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["evaluation_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
    )

    metrics_logger = MetricsLoggerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[metrics_logger]
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    print("Training Done!!")
    # --------------------------------------Evaluate on Validation dataset---------------------------------------------#

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    print(f"Perplexity: {perplexity}")
    print("Evaluation Done!!")

    # ------------------------------------------Test on new sample-----------------------------------------------------#

    inputs = tokenizer(gen_cfg["test_prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"]
        )
    print("Generated Text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Testing Done!!")

    # ------------------------------------------Log experiment-----------------------------------------------------#
    train_metrics = {
        "epoch_times": metrics_logger.epoch_times,
        "epoch_losses": metrics_logger.epoch_losses
    }
    log_experiment(config, eval_results, train_metrics, output_dir=train_cfg["results_logging"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file for hyper-parameters")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
