"""GradCAM visualization for 3D CNN."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Optional, Tuple
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class VideoGradCAM:
    """GradCAM wrapper for 3D CNN video classification."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        device: torch.device,
    ):
        """
        Initialize VideoGradCAM.

        Args:
            model: PyTorch model
            target_layers: List of layers to compute GradCAM on
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.target_layers = target_layers

        # Initialize GradCAM
        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers,
        )

    def generate_heatmap(
        self,
        input_clip: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for video clip.

        Args:
            input_clip: Input video clip (1, C, T, H, W) or (C, T, H, W)
            target_class: Target class for GradCAM. If None, uses predicted class

        Returns:
            Heatmap of shape (T, H, W) with values in [0, 1]
        """
        # Ensure batch dimension
        if input_clip.dim() == 4:
            input_clip = input_clip.unsqueeze(0)

        input_clip = input_clip.to(self.device)

        # Get target
        if target_class is None:
            # Use predicted class
            with torch.no_grad():
                output = self.model(input_clip)
                target_class = output.argmax(dim=1).item()

        targets = [ClassifierOutputTarget(target_class)]

        # Generate CAM
        # Note: GradCAM returns (B, T, H, W) for 3D inputs
        grayscale_cam = self.cam(input_tensor=input_clip, targets=targets)

        # Remove batch dimension
        grayscale_cam = grayscale_cam[0]  # (T, H, W)

        return grayscale_cam

    def overlay_heatmap_on_frame(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on frame.

        Args:
            frame: RGB frame (H, W, 3) in range [0, 255]
            heatmap: Heatmap (H, W) in range [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap

        Returns:
            Overlayed frame (H, W, 3)
        """
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Resize heatmap to match frame size if needed
        if heatmap.shape != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Convert heatmap to colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Convert from BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

        return overlayed

    def visualize_clip(
        self,
        input_clip: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
        denormalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize GradCAM for entire clip.

        Args:
            input_clip: Input video clip (1, C, T, H, W) or (C, T, H, W)
            target_class: Target class for GradCAM
            alpha: Blending factor
            colormap: OpenCV colormap
            denormalize: If True, denormalize input clip
            mean: Normalization mean (per channel)
            std: Normalization std (per channel)

        Returns:
            Tuple of (original_frames, heatmap_frames, overlayed_frames)
            Each is np.ndarray of shape (T, H, W, 3)
        """
        # Ensure batch dimension
        if input_clip.dim() == 4:
            input_clip_batched = input_clip.unsqueeze(0)
        else:
            input_clip_batched = input_clip

        # Generate heatmap
        heatmap_3d = self.generate_heatmap(input_clip_batched, target_class)  # (T, H, W)

        # Get original frames
        if input_clip.dim() == 5:
            clip = input_clip[0]  # Remove batch dim
        else:
            clip = input_clip

        # Denormalize if needed
        if denormalize:
            clip = clip.clone()
            mean_tensor = torch.tensor(mean).view(3, 1, 1, 1).to(clip.device)
            std_tensor = torch.tensor(std).view(3, 1, 1, 1).to(clip.device)
            clip = clip * std_tensor + mean_tensor
            clip = torch.clamp(clip, 0, 1)

        # Convert to numpy: (C, T, H, W) -> (T, H, W, C)
        clip_np = clip.permute(1, 2, 3, 0).cpu().numpy()
        clip_np = (clip_np * 255).astype(np.uint8)

        num_frames = clip_np.shape[0]

        # Process each frame
        heatmap_frames = []
        overlayed_frames = []

        for t in range(num_frames):
            frame = clip_np[t]
            heatmap = heatmap_3d[t]

            # Create heatmap visualization
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            heatmap_frames.append(heatmap_colored)

            # Create overlay
            overlayed = self.overlay_heatmap_on_frame(frame, heatmap, alpha, colormap)
            overlayed_frames.append(overlayed)

        original_frames = clip_np
        heatmap_frames = np.array(heatmap_frames)
        overlayed_frames = np.array(overlayed_frames)

        return original_frames, heatmap_frames, overlayed_frames

    def save_frames_as_video(
        self,
        frames: np.ndarray,
        save_path: str,
        fps: int = 10,
    ):
        """
        Save frames as video.

        Args:
            frames: Frames (T, H, W, 3) in range [0, 255]
            save_path: Output path
            fps: Frames per second
        """
        num_frames, height, width, channels = frames.shape

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def create_side_by_side_video(
        self,
        original_frames: np.ndarray,
        overlayed_frames: np.ndarray,
        save_path: str,
        fps: int = 10,
    ):
        """
        Create side-by-side video of original and GradCAM overlay.

        Args:
            original_frames: Original frames (T, H, W, 3)
            overlayed_frames: Overlayed frames (T, H, W, 3)
            save_path: Output path
            fps: Frames per second
        """
        # Concatenate horizontally
        combined_frames = np.concatenate([original_frames, overlayed_frames], axis=2)

        self.save_frames_as_video(combined_frames, save_path, fps)


def get_target_layers(model: nn.Module, layer_name: str = "block4") -> List[nn.Module]:
    """
    Get target layers for GradCAM.

    Args:
        model: PyTorch model
        layer_name: Name of block to target (block3=14x14, block4=7x7, block5=3x3 spatial)

    Returns:
        List of target layers
    """
    if layer_name == "block5":
        return [model.block5[-2]]  # 3x3 spatial - very coarse
    elif layer_name == "block4":
        return [model.block4[-2]]  # 7x7 spatial - good balance (default)
    elif layer_name == "block3":
        return [model.block3[-2]]  # 14x14 spatial - best detail
    else:
        raise ValueError(f"Unknown layer name: {layer_name}")
