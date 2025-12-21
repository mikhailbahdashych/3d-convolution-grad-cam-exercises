"""Main training script for exercise recognition."""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logger
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.cnn3d import build_c3d, count_parameters
from src.training.losses import build_loss_fn
from src.training.trainer import Trainer


def create_data_loaders(config):
    """Create training and validation data loaders."""
    logger = setup_logger("data")

    # Create datasets
    logger.info("Creating datasets...")

    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_dataset = ExerciseVideoDataset(
        data_root=config.get("data.root_dir"),
        split="train",
        clip_length=config.get("preprocessing.clip_length"),
        temporal_stride=config.get("preprocessing.temporal_stride"),
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=train_transform,
        filter_background=True,
        cache_videos=True,  # Cache videos in RAM for much faster loading
    )

    val_dataset = ExerciseVideoDataset(
        data_root=config.get("data.root_dir"),
        split="test",
        clip_length=config.get("preprocessing.clip_length"),
        temporal_stride=16,  # No overlap for validation
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=val_transform,
        filter_background=True,
        cache_videos=True,  # Cache videos in RAM for much faster loading
    )

    logger.info(f"Train dataset: {len(train_dataset)} clips")
    logger.info(f"Val dataset: {len(val_dataset)} clips")

    # Create samplers for handling class imbalance
    if config.get("sampling.use_weighted_sampler", True):
        logger.info("Using weighted random sampler for class balance")
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("training.batch_size"),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.get("training.num_workers"),
        pin_memory=config.get("training.pin_memory"),
        persistent_workers=config.get("training.persistent_workers", True),
        prefetch_factor=config.get("training.prefetch_factor", 2),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("evaluation.batch_size"),
        shuffle=False,
        num_workers=config.get("training.num_workers"),
        pin_memory=config.get("training.pin_memory"),
        persistent_workers=config.get("training.persistent_workers", True),
        prefetch_factor=config.get("training.prefetch_factor", 2),
    )

    # Get class weights for loss function
    class_weights = train_dataset.get_class_weights()

    return train_loader, val_loader, class_weights


def create_model(config, device):
    """Create model."""
    logger = setup_logger("model")

    logger.info("Creating model...")
    model = build_c3d(config)

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    return model


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    logger = setup_logger("optimizer")

    # Optimizer
    optimizer_name = config.get("training.optimizer.name", "AdamW")
    lr = config.get("training.optimizer.lr")
    weight_decay = config.get("training.optimizer.weight_decay")

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    logger.info(f"Optimizer: {optimizer_name}, lr={lr}, weight_decay={weight_decay}")

    # Scheduler
    scheduler_name = config.get("training.scheduler.name")

    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("training.scheduler.mode", "min"),
            factor=config.get("training.scheduler.factor", 0.5),
            patience=config.get("training.scheduler.patience", 5),
            min_lr=config.get("training.scheduler.min_lr", 1e-7),
        )
        logger.info(f"Scheduler: ReduceLROnPlateau")
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("training.num_epochs"),
        )
        logger.info(f"Scheduler: CosineAnnealingLR")
    else:
        scheduler = None
        logger.info("No learning rate scheduler")

    return optimizer, scheduler


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train exercise recognition model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.seed is not None:
        config.config["seed"] = args.seed
    if args.epochs is not None:
        config.config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config.config["training"]["batch_size"] = args.batch_size

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup logger
    logger = setup_logger("main", config.get("logging.log_dir"))
    logger.info("=" * 60)
    logger.info("EXERCISE RECOGNITION TRAINING")
    logger.info("=" * 60)

    # Get device
    if args.device:
        device = torch.device(args.device)
        logger.info(f"Using device: {device} (from command line)")
    else:
        device = get_device(
            prefer_cuda=config.get("device.use_cuda", True),
            prefer_mps=config.get("device.use_mps", True),
        )

    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(config)

    # Create model
    model = create_model(config, device)

    # Create loss function
    logger.info("Creating loss function...")
    class_weights = class_weights.to(device)
    criterion = build_loss_fn(config, class_weights)
    logger.info(f"Loss function: {type(criterion).__name__}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scheduler=scheduler,
        logger=logger,
    )

    # Start training
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    trainer.train()

    # Print final metrics
    metrics = trainer.get_metrics()
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.2f}%")
    logger.info(f"Checkpoints saved to: {config.get('checkpoint.save_dir')}")


if __name__ == "__main__":
    main()
