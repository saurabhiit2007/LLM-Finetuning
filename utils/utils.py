import csv
import os
from datetime import datetime
import torch
import psutil
from transformers import TrainerCallback, TrainerState, TrainerControl
import time


def get_gpu_memory():
    """Returns total allocated GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / (1024 * 1024)
    return None


def log_experiment(config, eval_results, train_metrics=None, output_dir="logs"):
    """
    Logs hyperparameters, evaluation results, training metrics, GPU usage, and sample text to a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment_log.csv")

    # Flatten config for logging
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_config[f"{k}.{kk}"] = vv
        else:
            flat_config[k] = v

    # Flatten evaluation metrics
    flat_eval = {}
    for k, v in eval_results.items():
        flat_eval[f"eval.{k}"] = v if not isinstance(v, torch.Tensor) else v.item()

    # Flatten training metrics (optional)
    flat_train = {}
    if train_metrics:
        for k, v in train_metrics.items():
            flat_train[f"train.{k}"] = v

    # Resource usage
    gpu_mem = get_gpu_memory()
    cpu_mem = psutil.virtual_memory().used / (1024 * 1024)
    flat_resource = {"gpu_memory_mb": gpu_mem, "cpu_memory_mb": cpu_mem}

    # Combine all logs
    log_row = {
        **flat_config,
        **flat_eval,
        **flat_train,
        **flat_resource,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write CSV
    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    print(f"[INFO] Experiment logged at {log_file}")


class MetricsLoggerCallback(TrainerCallback):
    """
    Custom callback to log per-epoch training time and loss.
    """
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
        self.epoch_losses = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        # average loss for epoch
        self.epoch_losses.append(state.log_history[-1]["loss"] if state.log_history else None)