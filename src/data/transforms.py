"""Video transformation and augmentation."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from typing import Optional, List, Tuple


class VideoNormalize:
    """Normalize video with mean and std."""

    def __init__(self, mean: List[float], std: List[float]):
        """
        Initialize normalization.

        Args:
            mean: Per-channel mean [R, G, B]
            std: Per-channel std [R, G, B]
        """
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Normalize clip.

        Args:
            clip: Tensor of shape (C, T, H, W)

        Returns:
            Normalized clip
        """
        return (clip - self.mean) / self.std


class VideoRandomHorizontalFlip:
    """Randomly flip video horizontally."""

    def __init__(self, p: float = 0.5):
        """
        Initialize flip transform.

        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply horizontal flip.

        Args:
            clip: Tensor of shape (C, T, H, W)

        Returns:
            Flipped clip
        """
        if torch.rand(1).item() < self.p:
            return torch.flip(clip, dims=[3])  # Flip width dimension
        return clip


class VideoRandomRotation:
    """Randomly rotate video frames."""

    def __init__(self, degrees: float):
        """
        Initialize rotation transform.

        Args:
            degrees: Range of rotation (-degrees, +degrees)
        """
        self.degrees = degrees

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation.

        Args:
            clip: Tensor of shape (C, T, H, W)

        Returns:
            Rotated clip
        """
        angle = torch.rand(1).item() * 2 * self.degrees - self.degrees

        # Apply same rotation to all frames
        C, T, H, W = clip.shape

        # Rotate each frame
        rotated_frames = []
        for t in range(T):
            frame = clip[:, t, :, :]  # (C, H, W)
            rotated = TF.rotate(frame, angle, fill=0)
            rotated_frames.append(rotated)

        # Stack back: (T, C, H, W) -> (C, T, H, W)
        rotated_clip = torch.stack(rotated_frames, dim=1)

        return rotated_clip


class VideoColorJitter:
    """Apply color jittering to video."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        """
        Initialize color jitter.

        Args:
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
        """
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply color jitter.

        Args:
            clip: Tensor of shape (C, T, H, W)

        Returns:
            Jittered clip
        """
        C, T, H, W = clip.shape

        # Apply same jitter to all frames
        # Reshape to (T, C, H, W) for torchvision
        clip_t = clip.permute(1, 0, 2, 3)

        jittered_frames = []
        for t in range(T):
            frame = clip_t[t]
            jittered = self.color_jitter(frame)
            jittered_frames.append(jittered)

        jittered_clip = torch.stack(jittered_frames, dim=0)

        # Reshape back to (C, T, H, W)
        return jittered_clip.permute(1, 0, 2, 3)


class VideoRandomCrop:
    """Randomly crop video frames."""

    def __init__(self, size: int, scale: Tuple[float, float] = (0.8, 1.0)):
        """
        Initialize random crop.

        Args:
            size: Output size (square)
            scale: Range of crop scale
        """
        self.size = size
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply random crop.

        Args:
            clip: Tensor of shape (C, T, H, W)

        Returns:
            Cropped clip
        """
        C, T, H, W = clip.shape

        # Determine crop size
        scale = torch.rand(1).item() * (self.scale[1] - self.scale[0]) + self.scale[0]
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        # Random crop position
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()

        # Crop all frames
        clip_cropped = clip[:, :, top:top+crop_h, left:left+crop_w]

        # Resize to target size
        clip_resized = F.interpolate(
            clip_cropped.view(C * T, 1, crop_h, crop_w),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False
        )

        return clip_resized.view(C, T, self.size, self.size)


class VideoCompose:
    """Compose multiple video transforms."""

    def __init__(self, transforms: List):
        """
        Initialize compose transform.

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms sequentially.

        Args:
            clip: Input clip

        Returns:
            Transformed clip
        """
        for transform in self.transforms:
            clip = transform(clip)
        return clip


def get_train_transforms(config) -> VideoCompose:
    """
    Get training transforms.

    Args:
        config: Configuration object

    Returns:
        VideoCompose with training transforms
    """
    transforms = []

    # Spatial augmentations
    if config.get("augmentation.spatial.horizontal_flip", 0) > 0:
        transforms.append(
            VideoRandomHorizontalFlip(p=config.get("augmentation.spatial.horizontal_flip"))
        )

    if config.get("augmentation.spatial.rotation_degrees", 0) > 0:
        transforms.append(
            VideoRandomRotation(degrees=config.get("augmentation.spatial.rotation_degrees"))
        )

    if config.get("augmentation.spatial.color_jitter"):
        jitter = config.get("augmentation.spatial.color_jitter")
        transforms.append(
            VideoColorJitter(
                brightness=jitter[0],
                contrast=jitter[1],
                saturation=jitter[2],
                hue=jitter[3],
            )
        )

    if config.get("augmentation.spatial.random_crop_scale"):
        scale = config.get("augmentation.spatial.random_crop_scale")
        spatial_size = config.get("preprocessing.spatial_size")
        # Note: We'll need to resize to a larger size first, then crop
        # For now, skip this as it's complex with our current pipeline

    # Normalization
    transforms.append(
        VideoNormalize(
            mean=config.get("normalization.mean"),
            std=config.get("normalization.std"),
        )
    )

    return VideoCompose(transforms)


def get_val_transforms(config) -> VideoCompose:
    """
    Get validation transforms (no augmentation).

    Args:
        config: Configuration object

    Returns:
        VideoCompose with validation transforms
    """
    transforms = [
        VideoNormalize(
            mean=config.get("normalization.mean"),
            std=config.get("normalization.std"),
        )
    ]

    return VideoCompose(transforms)
