"""Configuration management for the project."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Configuration class for loading and accessing config parameters."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or return default config."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = self._get_default_config()

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        project_root = Path(__file__).parent.parent.parent
        dataset_root = project_root / "dataset"

        return {
            "data": {
                "root_dir": str(dataset_root),
                "video_dir": "dataset/anon",
                "label_dir": "label",
                "skeleton_dir": "dataset/skeleton/yolo_pose_csv",
                "mask_dir": "dataset/mask",
                "split_file": "split.csv",
            },
            "preprocessing": {
                "clip_length": 16,
                "temporal_stride": 8,
                "spatial_size": 112,
                "fps": None,
            },
            "augmentation": {
                "spatial": {
                    "horizontal_flip": 0.5,
                    "rotation_degrees": 0,  # Disabled for speed (can re-enable later)
                    "color_jitter": [0.2, 0.2, 0.2, 0.1],
                    "random_crop_scale": [0.8, 1.0],
                },
                "temporal": {
                    "temporal_jitter": True,
                },
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "name": "C3D",
                "num_classes": 17,
                "dropout": 0.5,
                "pretrained": False,
            },
            "training": {
                "batch_size": 8,
                "num_epochs": 100,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
                "optimizer": {
                    "name": "AdamW",
                    "lr": 0.0001,
                    "weight_decay": 0.00001,
                },
                "scheduler": {
                    "name": "ReduceLROnPlateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 0.0000001,
                },
                "loss": {
                    "name": "FocalLoss",
                    "focal_gamma": 2.0,
                    "class_weights": "auto",
                },
                "early_stopping": {
                    "patience": 10,
                    "min_delta": 0.001,
                },
                "gradient_clipping": {
                    "max_norm": 1.0,
                },
            },
            "sampling": {
                "use_weighted_sampler": True,
                "oversampling_factor": 1.0,
            },
            "device": {
                "use_cuda": True,
                "use_mps": True,
                "fallback_cpu": True,
            },
            "checkpoint": {
                "save_dir": "outputs/checkpoints",
                "save_freq": 10,
                "keep_best_n": 3,
            },
            "logging": {
                "log_dir": "outputs/logs",
                "tensorboard": True,
                "wandb": False,
                "log_freq": 10,
            },
            "evaluation": {
                "batch_size": 16,
                "save_predictions": True,
                "confusion_matrix": True,
                "per_class_metrics": True,
            },
            "gradcam": {
                "target_layer": "block5.conv2",
                "save_dir": "outputs/visualizations",
                "num_samples": 20,
                "overlay_alpha": 0.5,
                "colormap": "jet",
            },
            "seed": 42,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'model.num_classes')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)

    def save(self, path: str):
        """
        Save configuration to YAML file.

        Args:
            path: Output path for config file
        """
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from file or use default.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    return Config(config_path)
