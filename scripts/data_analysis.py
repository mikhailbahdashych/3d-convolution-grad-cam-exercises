"""Dataset analysis script."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import load_config


def analyze_split(config):
    """Analyze train/test split."""
    split_file = Path(config.get("data.root_dir")) / config.get("data.split_file")
    df = pd.read_csv(split_file)

    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT ANALYSIS")
    print("=" * 60)
    print(f"\nTotal subjects: {len(df)}")
    print(f"Train subjects: {(df['split'] == 'train').sum()}")
    print(f"Test subjects: {(df['split'] == 'test').sum()}")

    return df


def analyze_labels(config, split_df):
    """Analyze label distribution."""
    label_dir = Path(config.get("data.root_dir")) / config.get("data.label_dir")
    label_files = sorted(label_dir.glob("*.csv"))

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)

    all_labels_train = []
    all_labels_test = []

    for label_file in label_files:
        subject_id = label_file.stem
        split = split_df[split_df['id'] == subject_id]['split'].values

        if len(split) == 0:
            continue

        split = split[0]

        # Read labels
        df = pd.read_csv(label_file, header=None, names=["frame", "col1", "col2"])

        # Extract exercise labels (col2), excluding -001 (background)
        labels = df["col2"].values
        exercise_labels = labels[labels != -1]

        if split == "train":
            all_labels_train.extend(exercise_labels)
        else:
            all_labels_test.extend(exercise_labels)

    # Count labels
    train_counts = Counter(all_labels_train)
    test_counts = Counter(all_labels_test)

    print(f"\nTotal training frames with exercises: {len(all_labels_train)}")
    print(f"Total test frames with exercises: {len(all_labels_test)}")

    print("\nClass distribution (Training):")
    for class_id in sorted(train_counts.keys()):
        print(f"  Class {class_id:2d}: {train_counts[class_id]:6d} frames")

    print("\nClass distribution (Test):")
    for class_id in sorted(test_counts.keys()):
        print(f"  Class {class_id:2d}: {test_counts[class_id]:6d} frames")

    return train_counts, test_counts


def analyze_videos(config, split_df):
    """Analyze video properties."""
    video_dir = Path(config.get("data.root_dir")) / config.get("data.video_dir")
    video_files = sorted(video_dir.glob("*.mp4"))[:5]  # Analyze first 5 videos

    print("\n" + "=" * 60)
    print("VIDEO PROPERTIES (sample of 5 videos)")
    print("=" * 60)

    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Could not open {video_file.name}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        print(f"\n{video_file.name}:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {duration:.2f}s")

        cap.release()


def plot_class_distribution(train_counts, test_counts, output_dir):
    """Plot class distribution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    all_classes = sorted(set(train_counts.keys()) | set(test_counts.keys()))
    train_values = [train_counts.get(c, 0) for c in all_classes]
    test_values = [test_counts.get(c, 0) for c in all_classes]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Training distribution
    axes[0].bar(all_classes, train_values, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Exercise Class')
    axes[0].set_ylabel('Number of Frames')
    axes[0].set_title('Training Set - Class Distribution')
    axes[0].set_xticks(all_classes)
    axes[0].grid(axis='y', alpha=0.3)

    # Test distribution
    axes[1].bar(all_classes, test_values, color='coral', alpha=0.8)
    axes[1].set_xlabel('Exercise Class')
    axes[1].set_ylabel('Number of Frames')
    axes[1].set_title('Test Set - Class Distribution')
    axes[1].set_xticks(all_classes)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nClass distribution plot saved to: {output_dir / 'class_distribution.png'}")


def main():
    """Main function."""
    config = load_config()

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Dataset root: {config.get('data.root_dir')}")

    # Analyze split
    split_df = analyze_split(config)

    # Analyze labels
    train_counts, test_counts = analyze_labels(config, split_df)

    # Analyze videos
    analyze_videos(config, split_df)

    # Plot distribution
    plot_class_distribution(train_counts, test_counts, "outputs/analysis")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
