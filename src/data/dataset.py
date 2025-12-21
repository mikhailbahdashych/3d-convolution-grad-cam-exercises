"""PyTorch Dataset for exercise video classification."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from collections import Counter

from .utils import load_video, extract_clip, load_labels, get_subject_id


class ExerciseVideoDataset(Dataset):
    """
    Dataset for exercise recognition from video clips.

    Loads video clips and corresponding labels for exercise classification.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        clip_length: int = 16,
        temporal_stride: int = 8,
        spatial_size: int = 112,
        transform=None,
        filter_background: bool = True,
        cache_videos: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory of dataset
            split: 'train' or 'test'
            clip_length: Number of frames per clip
            temporal_stride: Stride for sliding window (smaller = more clips)
            spatial_size: Target spatial resolution (height and width)
            transform: Optional transform to apply to clips
            filter_background: If True, only include clips with exercise labels (not -1)
            cache_videos: If True, cache all videos in memory (requires lots of RAM)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.clip_length = clip_length
        self.temporal_stride = temporal_stride
        self.spatial_size = spatial_size
        self.transform = transform
        self.filter_background = filter_background
        self.cache_videos = cache_videos

        # Paths
        self.video_dir = self.data_root / "dataset" / "anon"
        self.label_dir = self.data_root / "label"
        self.split_file = self.data_root / "split.csv"

        # Load split info
        self.split_df = pd.read_csv(self.split_file)
        self.subject_ids = self.split_df[self.split_df["split"] == split]["id"].tolist()

        # Build clip index
        self.clips = []
        self.label_counts = Counter()
        self._build_clip_index()

        # Cache for videos
        self.video_cache = {} if cache_videos else None
        if cache_videos:
            self._cache_all_videos()

        print(f"Loaded {len(self.clips)} clips for {split} split")
        print(f"Label distribution: {dict(self.label_counts)}")

    def _build_clip_index(self):
        """Build index of all valid clips."""
        for subject_id in self.subject_ids:
            video_path = self.video_dir / f"{subject_id}.mp4"
            label_path = self.label_dir / f"{subject_id}.csv"

            if not video_path.exists() or not label_path.exists():
                print(f"Warning: Missing data for {subject_id}, skipping")
                continue

            # Load labels
            labels = load_labels(str(label_path))

            # Determine number of clips we can extract
            # We'll use a sliding window approach
            num_frames = len(labels)

            # Generate clip start indices
            clip_starts = list(range(0, num_frames - self.clip_length + 1, self.temporal_stride))

            # For each potential clip, determine label and add to index
            for start_idx in clip_starts:
                end_idx = start_idx + self.clip_length

                # Get labels for this clip
                clip_labels = labels[start_idx:end_idx]

                # Determine clip label (majority voting, excluding background -1)
                exercise_labels = clip_labels[clip_labels != -1]

                if len(exercise_labels) == 0:
                    # All frames are background
                    if not self.filter_background:
                        # Assign a special background class (we'll map -1 to 0 later)
                        label = -1
                    else:
                        continue  # Skip background clips
                else:
                    # Use majority vote
                    label_counts = Counter(exercise_labels)
                    label = label_counts.most_common(1)[0][0]

                # Add clip to index
                self.clips.append({
                    "subject_id": subject_id,
                    "video_path": str(video_path),
                    "label_path": str(label_path),
                    "start_idx": start_idx,
                    "label": int(label),
                })

                # Update label counts
                if label != -1:
                    self.label_counts[int(label)] += 1

    def _cache_all_videos(self):
        """Cache all videos in memory."""
        print("Caching all videos in memory...")
        unique_videos = set(clip["video_path"] for clip in self.clips)

        for video_path in unique_videos:
            frames = load_video(video_path, target_size=(self.spatial_size, self.spatial_size))
            self.video_cache[video_path] = frames

        print(f"Cached {len(self.video_cache)} videos")

    def __len__(self) -> int:
        """Return number of clips in dataset."""
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a single clip.

        Args:
            idx: Clip index

        Returns:
            clip: Tensor of shape (C, T, H, W)
            label: Integer label
            metadata: Dictionary with additional info
        """
        clip_info = self.clips[idx]

        # Load video frames
        if self.video_cache and clip_info["video_path"] in self.video_cache:
            frames = self.video_cache[clip_info["video_path"]]
        else:
            frames = load_video(
                clip_info["video_path"],
                target_size=(self.spatial_size, self.spatial_size)
            )

        # Extract clip
        temporal_jitter = self.split == "train"  # Only jitter during training
        clip = extract_clip(
            frames,
            clip_info["start_idx"],
            self.clip_length,
            temporal_jitter=temporal_jitter,
        )

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float()

        # Normalize to [0, 1]
        clip = clip / 255.0

        # Apply transforms
        if self.transform:
            clip = self.transform(clip)

        # Get label (map -1 to 0 for background, shift others if needed)
        label = clip_info["label"]

        # Metadata
        metadata = {
            "subject_id": clip_info["subject_id"],
            "start_idx": clip_info["start_idx"],
        }

        return clip, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance.

        Returns:
            torch.Tensor: Class weights
        """
        # Get unique labels and their counts
        labels = [clip["label"] for clip in self.clips if clip["label"] != -1]

        if len(labels) == 0:
            return torch.ones(17)

        label_counts = Counter(labels)
        num_classes = max(label_counts.keys()) + 1

        total_samples = len(labels)
        weights = torch.zeros(num_classes)

        for class_id, count in label_counts.items():
            weights[class_id] = total_samples / (num_classes * count)

        return weights

    def get_sample_weights(self) -> List[float]:
        """
        Get per-sample weights for WeightedRandomSampler.

        Returns:
            List of sample weights
        """
        class_weights = self.get_class_weights()

        sample_weights = []
        for clip in self.clips:
            label = clip["label"]
            if label == -1:
                sample_weights.append(0.0)  # Don't sample background
            else:
                sample_weights.append(class_weights[label].item())

        return sample_weights
