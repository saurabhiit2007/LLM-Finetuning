import csv
import os
from datetime import datetime
import torch
import psutil
from transformers import TrainerCallback, TrainerState, TrainerControl
import time


def get_gpu_memory():
    """Returns GPU memory usage info (current + peak in MB)."""
    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    current = torch.cuda.memory_allocated(0) / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    return current, peak


def get_cpu_memory():
    """Returns process-specific CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def count_trainable_params(model):
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_experiment(config, eval_results, train_metrics=None, output_dir="logs"):
    """Logs hyperparameters, evaluation results, and summary metrics."""
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
    gpu_cur, gpu_peak = get_gpu_memory()
    cpu_mem = get_cpu_memory()
    flat_resource = {"gpu_current_mb": gpu_cur, "gpu_peak_mb": gpu_peak, "cpu_memory_mb": cpu_mem}

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
    Logs per-epoch timing, loss, and resource usage.
    """

    def __init__(self, model=None, output_dir="logs"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.epoch_start_time = None
        self.csv_file = os.path.join(output_dir, "epoch_metrics.csv")

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "epoch_time_sec",
                    "gpu_current_mb", "gpu_peak_mb", "cpu_memory_mb",
                    "total_params", "trainable_params", "timestamp"
                ])

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.synchronize()
        epoch_time = time.time() - self.epoch_start_time
        gpu_cur, gpu_peak = get_gpu_memory()
        cpu_mem = get_cpu_memory()
        loss = state.log_history[-1].get("loss", None) if state.log_history else None
        total_params, trainable_params = count_trainable_params(self.model)

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                state.epoch, loss, epoch_time,
                gpu_cur, gpu_peak, cpu_mem,
                total_params, trainable_params,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        print(
            f"[Epoch {state.epoch}] time={epoch_time:.2f}s | "
            f"loss={loss:.4f if loss else 'N/A'} | "
            f"GPU peak={gpu_peak:.2f} MB | CPU={cpu_mem:.2f} MB | "
            f"trainable={trainable_params / 1e6:.2f}M/{total_params / 1e6:.2f}M"
        )
