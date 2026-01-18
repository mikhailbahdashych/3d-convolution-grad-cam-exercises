"""Model checkpointing utilities."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import glob


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    filename: str = "checkpoint.pth",
    config: Optional[Dict[str, Any]] = None,
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics (e.g., {'val_loss': 0.5, 'val_acc': 0.9})
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        config: Optional configuration dict
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    # Handle DataParallel models - save the underlying model
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if config:
        checkpoint["config"] = config

    torch.save(checkpoint, save_path)
    return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional PyTorch optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Dictionary with checkpoint info (epoch, metrics, config)
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DataParallel models - load into the underlying model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }

    return info


def get_best_checkpoint(save_dir: str, metric: str = "val_acc", mode: str = "max") -> Optional[str]:
    """
    Find best checkpoint based on metric.

    Args:
        save_dir: Directory containing checkpoints
        metric: Metric name to compare
        mode: 'max' for metrics to maximize, 'min' for metrics to minimize

    Returns:
        Path to best checkpoint or None if no checkpoints found
    """
    checkpoint_files = glob.glob(str(Path(save_dir) / "*.pth"))

    if not checkpoint_files:
        return None

    best_checkpoint = None
    best_value = float("-inf") if mode == "max" else float("inf")

    for checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metrics = checkpoint.get("metrics", {})

            if metric in metrics:
                value = metrics[metric]

                if mode == "max" and value > best_value:
                    best_value = value
                    best_checkpoint = checkpoint_path
                elif mode == "min" and value < best_value:
                    best_value = value
                    best_checkpoint = checkpoint_path

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            continue

    return best_checkpoint


def cleanup_checkpoints(save_dir: str, keep_best_n: int = 3, metric: str = "val_acc", mode: str = "max"):
    """
    Remove old checkpoints, keeping only the best N.

    Args:
        save_dir: Directory containing checkpoints
        keep_best_n: Number of best checkpoints to keep
        metric: Metric name to rank checkpoints
        mode: 'max' or 'min' for ranking
    """
    checkpoint_files = glob.glob(str(Path(save_dir) / "checkpoint_epoch_*.pth"))

    if len(checkpoint_files) <= keep_best_n:
        return

    # Load and rank checkpoints
    checkpoints_with_metrics = []
    for checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metrics = checkpoint.get("metrics", {})
            if metric in metrics:
                checkpoints_with_metrics.append((checkpoint_path, metrics[metric]))
        except Exception:
            continue

    # Sort by metric
    reverse = mode == "max"
    checkpoints_with_metrics.sort(key=lambda x: x[1], reverse=reverse)

    # Remove checkpoints beyond keep_best_n
    for checkpoint_path, _ in checkpoints_with_metrics[keep_best_n:]:
        Path(checkpoint_path).unlink()
