"""GradCAM visualization script."""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.utils.device import get_device
from src.utils.checkpointing import load_checkpoint
from src.utils.logging import setup_logger
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_val_transforms
from src.data.utils import load_video, load_labels
from src.models.cnn3d import build_c3d
from src.visualization.gradcam import VideoGradCAM, get_target_layers


def process_long_sample(
    video_path: str,
    label_path: str,
    start_frame: int,
    num_frames: int,
    model: torch.nn.Module,
    gradcam: VideoGradCAM,
    config,
    device: torch.device,
    save_dir: Path,
    alpha: float = 0.5,
    fps: int = 30,
    scale_factor: int = 3,
    use_masks: bool = False,
    mask_alpha: float = 1.0,
    mask_dir: Path = None,
    subject_id: str = None,
    logger=None,
):
    """
    Process a long video sample with dynamic probability updates.

    Args:
        video_path: Path to video file
        label_path: Path to label CSV file
        start_frame: Starting frame index
        num_frames: Number of frames to process
        model: PyTorch model
        gradcam: VideoGradCAM instance
        config: Configuration object
        device: Device to run on
        save_dir: Directory to save outputs
        alpha: GradCAM overlay alpha
        fps: Output video FPS
        scale_factor: Scale factor for output
        use_masks: Whether to apply segmentation masks
        mask_alpha: Mask strength (1.0=full mask, 0.7=30% background visible)
        mask_dir: Directory containing masks
        subject_id: Subject ID for mask lookup
        logger: Logger instance
    """
    clip_length = config.get("preprocessing.clip_length")
    spatial_size = config.get("preprocessing.spatial_size")
    mean = config.get("normalization.mean")
    std = config.get("normalization.std")

    # Load video frames
    if logger:
        logger.info(f"Loading video: {video_path}")
    frames = load_video(video_path, target_size=(spatial_size, spatial_size))
    labels = load_labels(label_path)

    # Ensure we don't go past video end
    end_frame = min(start_frame + num_frames, len(frames) - clip_length)
    if end_frame <= start_frame:
        if logger:
            logger.warning("Not enough frames in video")
        return

    actual_frames = end_frame - start_frame
    if logger:
        logger.info(f"Processing frames {start_frame} to {end_frame} ({actual_frames} frames)")

    # Prepare outputs
    all_original_frames = []
    all_overlay_frames = []
    all_probabilities = []
    all_true_labels = []

    # Process with sliding window
    model.eval()

    for frame_idx in range(start_frame, end_frame):
        # Extract clip starting at this frame
        clip_start = frame_idx
        clip_end = clip_start + clip_length

        if clip_end > len(frames):
            break

        # Get clip frames
        clip = frames[clip_start:clip_end].copy()  # (T, H, W, C)

        # Apply soft masks if enabled
        if use_masks and mask_dir is not None and subject_id is not None:
            mask_subject_dir = mask_dir / subject_id
            if mask_subject_dir.exists():
                for i in range(clip_length):
                    mask_frame_idx = clip_start + i + 1  # 1-indexed
                    mask_path = mask_subject_dir / f"{mask_frame_idx:05d}_mask.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (spatial_size, spatial_size))
                        binary_mask = (mask > 127).astype(np.float32)
                        # Apply soft mask: person 100%, background (1-mask_alpha)%
                        soft_mask = mask_alpha * binary_mask[:, :, np.newaxis] + (1 - mask_alpha)
                        clip[i] = clip[i] * soft_mask

        # Store original first frame of this window
        all_original_frames.append(clip[0].copy())

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0

        # Apply normalization
        mean_tensor = torch.tensor(mean).view(3, 1, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean_tensor) / std_tensor

        # Add batch dimension
        clip_tensor = clip_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(clip_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        all_probabilities.append(probs)

        # Get true label for this frame
        true_label = int(labels[frame_idx]) if frame_idx < len(labels) else -1
        all_true_labels.append(true_label)

        # Generate GradCAM for this clip
        _, _, overlayed = gradcam.visualize_clip(
            input_clip=clip_tensor,
            target_class=None,
            alpha=alpha,
            denormalize=True,
            mean=mean,
            std=std,
        )

        # Store first frame of overlay (corresponds to current frame_idx)
        all_overlay_frames.append(overlayed[0])

        if logger and (frame_idx - start_frame) % 50 == 0:
            logger.info(f"  Processed frame {frame_idx - start_frame}/{actual_frames}")

    # Convert to arrays
    all_original_frames = np.array(all_original_frames)
    all_overlay_frames = np.array(all_overlay_frames)

    if logger:
        logger.info(f"Generated {len(all_original_frames)} frames")

    # Save original video
    gradcam.save_frames_as_video(
        all_original_frames,
        str(save_dir / "long_original.mp4"),
        fps=fps,
    )

    # Save overlay video
    gradcam.save_frames_as_video(
        all_overlay_frames,
        str(save_dir / "long_overlay.mp4"),
        fps=fps,
    )

    # Create side-by-side
    combined_frames = np.concatenate([all_original_frames, all_overlay_frames], axis=2)
    gradcam.save_frames_as_video(
        combined_frames,
        str(save_dir / "long_side_by_side.mp4"),
        fps=fps,
    )

    # Create visualization with dynamic probability chart
    overlay_with_dynamic_chart = gradcam.create_visualization_with_dynamic_chart(
        frames=all_overlay_frames,
        probabilities_per_frame=all_probabilities,
        true_labels_per_frame=all_true_labels,
        chart_width=250,
        scale_factor=scale_factor,
    )
    gradcam.save_frames_as_video(
        overlay_with_dynamic_chart,
        str(save_dir / "long_overlay_with_probs.mp4"),
        fps=fps,
    )

    # Save metadata
    metadata_file = save_dir / "long_sample_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Start Frame: {start_frame}\n")
        f.write(f"End Frame: {end_frame}\n")
        f.write(f"Total Frames: {actual_frames}\n")
        f.write(f"Duration (at {fps} fps): {actual_frames / fps:.2f} seconds\n")

    if logger:
        logger.info(f"Long sample visualization saved to: {save_dir}")


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
    parser.add_argument("--use-masks", action="store_true", help="Apply segmentation masks to remove background")
    parser.add_argument("--mask-alpha", type=float, default=1.0, help="Mask strength: 1.0=full mask, 0.7=30%% background visible")
    parser.add_argument("--scale-factor", type=int, default=3, help="Scale factor for overlay_with_probs video (default: 3, output will be 336x336 + chart)")
    parser.add_argument("--long-sample", action="store_true", help="Generate longer video samples with dynamic probability updates")
    parser.add_argument("--sample-duration", type=int, default=300, help="Duration of long samples in frames (default: 300 = 10 seconds at 30fps)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame for long sample mode (default: 0)")
    parser.add_argument("--subject-id", type=str, default=None, help="Specific subject ID to visualize in long sample mode")
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
        use_masks=args.use_masks,
        mask_alpha=args.mask_alpha,
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

    # Handle long sample mode
    if args.long_sample:
        logger.info("="*60)
        logger.info("LONG SAMPLE MODE")
        logger.info("="*60)
        logger.info(f"Duration: {args.sample_duration} frames")
        logger.info(f"Start frame: {args.start_frame}")

        # Get subject IDs
        data_root = Path(config.get("data.root_dir"))
        mask_dir = data_root / "dataset" / "mask"

        if args.subject_id:
            # Use specified subject
            subject_ids = [args.subject_id]
            logger.info(f"Using specified subject: {args.subject_id}")
        else:
            # Get from split file
            split_file = data_root / "split.csv"
            split_df = pd.read_csv(split_file)
            subject_ids = split_df[split_df["split"] == args.split]["id"].tolist()

        if not subject_ids:
            logger.error(f"No subjects found for split: {args.split}")
            return

        # Process requested number of samples
        for sample_idx in range(min(args.num_samples, len(subject_ids))):
            subject_id = subject_ids[sample_idx]

            video_path = data_root / "dataset" / "anon" / f"{subject_id}.mp4"
            label_path = data_root / "label" / f"{subject_id}.csv"

            if not video_path.exists() or not label_path.exists():
                logger.warning(f"Missing data for {subject_id}, skipping")
                continue

            sample_save_dir = save_dir / f"long_sample_{sample_idx:04d}_{subject_id}_f{args.start_frame}"
            sample_save_dir.mkdir(exist_ok=True)

            logger.info(f"\nProcessing long sample {sample_idx + 1}/{args.num_samples}: {subject_id}")

            process_long_sample(
                video_path=str(video_path),
                label_path=str(label_path),
                start_frame=args.start_frame,
                num_frames=args.sample_duration,
                model=model,
                gradcam=gradcam,
                config=config,
                device=device,
                save_dir=sample_save_dir,
                alpha=args.alpha,
                fps=args.fps,
                scale_factor=args.scale_factor,
                use_masks=args.use_masks,
                mask_alpha=args.mask_alpha,
                mask_dir=mask_dir,
                subject_id=subject_id,
                logger=logger,
            )

        logger.info("\n" + "="*60)
        logger.info("LONG SAMPLE VISUALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Visualizations saved to: {save_dir}")
        return

    # Process samples (standard mode)
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

        # Get prediction and probabilities
        with torch.no_grad():
            outputs = model(clips)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]  # (num_classes,)
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

        # Create visualization with probability chart
        overlay_with_chart = gradcam.create_visualization_with_chart(
            frames=overlayed_frames,
            probabilities=probabilities,
            true_label=true_label,
            predicted_label=predicted,
            chart_width=250,
            scale_factor=args.scale_factor,
        )
        gradcam.save_frames_as_video(
            overlay_with_chart,
            str(sample_dir / "overlay_with_probs.mp4"),
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
            f"{'+' if is_correct else '-'} -> {sample_dir.name}"
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
