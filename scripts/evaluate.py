"""Evaluation script."""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.utils.device import get_device
from src.utils.checkpointing import load_checkpoint
from src.utils.logging import setup_logger
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_val_transforms
from src.models.cnn3d import build_c3d
from src.evaluation.evaluator import Evaluator


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate exercise recognition model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--save-dir", type=str, default="outputs/results", help="Directory to save results")
    parser.add_argument("--use-masks", action="store_true", help="Apply segmentation masks to remove background")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override batch size
    if args.batch_size:
        config.config["evaluation"]["batch_size"] = args.batch_size

    # Setup logger
    logger = setup_logger("evaluate")
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    logger.info(f"Device: {device}")

    # Create dataset
    logger.info("Loading dataset...")
    val_transform = get_val_transforms(config)

    dataset = ExerciseVideoDataset(
        data_root=config.get("data.root_dir"),
        split=args.split,
        clip_length=config.get("preprocessing.clip_length"),
        temporal_stride=16,  # No overlap for evaluation
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=val_transform,
        filter_background=True,
        cache_videos=True,
        use_masks=args.use_masks,
    )

    logger.info(f"Dataset size: {len(dataset)} clips")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config.get("evaluation.batch_size"),
        shuffle=False,
        num_workers=config.get("training.num_workers", 4),
        pin_memory=config.get("training.pin_memory", True),
    )

    # Create model
    logger.info("Creating model...")
    model = build_c3d(config)

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint_info = load_checkpoint(args.checkpoint, model, device=device)

    logger.info(f"Checkpoint epoch: {checkpoint_info['epoch']}")
    if checkpoint_info['metrics']:
        logger.info(f"Checkpoint metrics: {checkpoint_info['metrics']}")

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(
        model=model,
        data_loader=data_loader,
        device=device,
        num_classes=config.get("model.num_classes", 17),
        save_dir=args.save_dir,
    )

    # Run evaluation
    logger.info("="*60)
    logger.info("STARTING EVALUATION")
    logger.info("="*60)

    metrics = evaluator.evaluate(save_predictions=True)

    # Analyze errors
    logger.info("\n" + "="*60)
    logger.info("ERROR ANALYSIS")
    logger.info("="*60)
    evaluator.analyze_errors()

    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
