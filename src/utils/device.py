"""Device selection utilities for CPU, MPS, and CUDA support."""

import torch


def get_device(prefer_cuda=True, prefer_mps=True, verbose=True):
    """
    Select the best available device for training.

    Priority: CUDA > MPS > CPU (if preferences enabled)

    Args:
        prefer_cuda: If True, prefer CUDA if available
        prefer_mps: If True, prefer MPS if available (and CUDA not available/preferred)
        verbose: If True, print selected device

    Returns:
        torch.device: Selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")

    return device


def get_device_info(device):
    """
    Get detailed information about the device.

    Args:
        device: torch.device

    Returns:
        dict: Device information
    """
    info = {
        "type": device.type,
        "index": device.index if device.type != "cpu" else None,
    }

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["memory_total"] = torch.cuda.get_device_properties(device).total_memory
        info["memory_available"] = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"

    return info


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
