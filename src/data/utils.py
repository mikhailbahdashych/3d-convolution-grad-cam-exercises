"""Data utility functions."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_video(video_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load video from file.

    Args:
        video_path: Path to video file
        target_size: Optional (height, width) to resize frames

    Returns:
        np.ndarray: Video frames with shape (T, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if target_size:
            frame = cv2.resize(frame, (target_size[1], target_size[0]))

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from video: {video_path}")

    return np.array(frames)


def extract_clip(
    frames: np.ndarray,
    start_idx: int,
    clip_length: int,
    temporal_jitter: bool = False,
) -> np.ndarray:
    """
    Extract a clip of specified length from video frames.

    Args:
        frames: Video frames with shape (T, H, W, C)
        start_idx: Starting frame index
        clip_length: Number of frames in clip
        temporal_jitter: If True, add random temporal jittering

    Returns:
        np.ndarray: Clip with shape (clip_length, H, W, C)
    """
    num_frames = len(frames)

    # Handle temporal jittering
    if temporal_jitter:
        # Add random offset within a small range
        jitter_range = min(4, num_frames - start_idx - clip_length)
        if jitter_range > 0:
            jitter = np.random.randint(-jitter_range, jitter_range + 1)
            start_idx = max(0, min(start_idx + jitter, num_frames - clip_length))

    # Extract clip
    end_idx = start_idx + clip_length

    # Handle boundary cases
    if end_idx > num_frames:
        # Pad with last frame if needed
        clip = frames[start_idx:]
        padding_needed = clip_length - len(clip)
        if padding_needed > 0:
            last_frame = frames[-1][np.newaxis, ...]
            padding = np.repeat(last_frame, padding_needed, axis=0)
            clip = np.concatenate([clip, padding], axis=0)
    else:
        clip = frames[start_idx:end_idx]

    return clip


def load_labels(label_path: str) -> np.ndarray:
    """
    Load labels from CSV file.

    Args:
        label_path: Path to label CSV file

    Returns:
        np.ndarray: Labels array
    """
    import pandas as pd

    df = pd.read_csv(label_path, header=None, names=["frame", "col1", "col2"])

    # Extract exercise labels from col2 (third column)
    # -001 = background/no exercise, 0000-0016 = exercise classes
    labels = df["col2"].values

    return labels


def get_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename.

    Args:
        filename: Filename (e.g., '001_d26arrd1.mp4')

    Returns:
        str: Subject ID (e.g., '001_d26arrd1')
    """
    return Path(filename).stem


def compute_class_weights(label_counts: dict, num_classes: int) -> np.ndarray:
    """
    Compute class weights for handling imbalance.

    Args:
        label_counts: Dictionary mapping class_id to count
        num_classes: Total number of classes

    Returns:
        np.ndarray: Class weights
    """
    total_samples = sum(label_counts.values())
    weights = np.zeros(num_classes)

    for class_id, count in label_counts.items():
        if count > 0:
            weights[class_id] = total_samples / (num_classes * count)

    return weights
