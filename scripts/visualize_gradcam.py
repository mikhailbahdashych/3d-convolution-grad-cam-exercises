"""GradCAM visualization script."""

import sys
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.utils.device import get_device
from src.utils.checkpointing import load_checkpoint
from src.utils.logging import setup_logger
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_val_transforms
from src.models.cnn3d import build_c3d
from src.visualization.gradcam import VideoGradCAM, get_target_layers


def main():
    """Main GradCAM visualization function."""
    parser = argparse.ArgumentParser(description="Visualize GradCAM for exercise recognition")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--save-dir", type=str, default="outputs/visualizations", help="Directory to save visualizations")
    parser.add_argument("--layer", type=str, default="block5", help="Target layer for GradCAM")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha (0-1)")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--misclassified-only", action="store_true", help="Only visualize misclassified samples")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger("gradcam")
    logger.info("="*60)
    logger.info("GRADCAM VISUALIZATION")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Target layer: {args.layer}")

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
        temporal_stride=16,
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=val_transform,
        filter_background=True,
        cache_videos=False,
    )

    logger.info(f"Dataset size: {len(dataset)} clips")

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time
        shuffle=False,
        num_workers=0,
    )

    # Create model
    logger.info("Creating model...")
    model = build_c3d(config)

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint_info = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Checkpoint epoch: {checkpoint_info['epoch']}")

    # Get target layers
    target_layers = get_target_layers(model, args.layer)

    # Create GradCAM
    logger.info("Creating GradCAM...")
    gradcam = VideoGradCAM(
        model=model,
        target_layers=target_layers,
        device=device,
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    logger.info("Generating GradCAM visualizations...")
    model.eval()

    num_processed = 0
    num_correct = 0
    num_incorrect = 0

    for batch_idx, (clips, labels, metadata) in enumerate(data_loader):
        if num_processed >= args.num_samples:
            break

        clips = clips.to(device)
        labels = labels.to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(clips)
            predicted = outputs.argmax(dim=1).item()

        true_label = labels.item()
        is_correct = predicted == true_label

        # Skip if only visualizing misclassified and this is correct
        if args.misclassified_only and is_correct:
            continue

        # Update counts
        if is_correct:
            num_correct += 1
        else:
            num_incorrect += 1

        # Generate GradCAM
        original_frames, heatmap_frames, overlayed_frames = gradcam.visualize_clip(
            input_clip=clips,
            target_class=None,  # Use predicted class
            alpha=args.alpha,
            denormalize=True,
            mean=config.get("normalization.mean"),
            std=config.get("normalization.std"),
        )

        # Create sample directory
        subject_id = metadata['subject_id'][0]
        start_idx = metadata['start_idx'][0].item()
        sample_dir = save_dir / f"sample_{num_processed:04d}_{subject_id}_{start_idx}"
        sample_dir.mkdir(exist_ok=True)

        # Save videos
        gradcam.save_frames_as_video(
            original_frames,
            str(sample_dir / "original.mp4"),
            fps=args.fps,
        )

        gradcam.save_frames_as_video(
            heatmap_frames,
            str(sample_dir / "heatmap.mp4"),
            fps=args.fps,
        )

        gradcam.save_frames_as_video(
            overlayed_frames,
            str(sample_dir / "overlay.mp4"),
            fps=args.fps,
        )

        gradcam.create_side_by_side_video(
            original_frames,
            overlayed_frames,
            str(sample_dir / "side_by_side.mp4"),
            fps=args.fps,
        )

        # Save metadata
        metadata_file = sample_dir / "metadata.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Subject ID: {subject_id}\n")
            f.write(f"Start Index: {start_idx}\n")
            f.write(f"True Label: {true_label}\n")
            f.write(f"Predicted Label: {predicted}\n")
            f.write(f"Correct: {is_correct}\n")

        logger.info(
            f"Sample {num_processed + 1}/{args.num_samples}: "
            f"True={true_label}, Pred={predicted}, "
            f"{'✓' if is_correct else '✗'} -> {sample_dir.name}"
        )

        num_processed += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples processed: {num_processed}")
    logger.info(f"Correct predictions: {num_correct}")
    logger.info(f"Incorrect predictions: {num_incorrect}")
    if num_processed > 0:
        logger.info(f"Accuracy: {100 * num_correct / num_processed:.2f}%")
    logger.info(f"Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
