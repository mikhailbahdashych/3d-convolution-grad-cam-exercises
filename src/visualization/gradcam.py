"""GradCAM visualization for 3D CNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO


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

    def create_probability_chart(
        self,
        probabilities: np.ndarray,
        true_label: int,
        predicted_label: int,
        height: int = 400,
        width: int = 300,
    ) -> np.ndarray:
        """
        Create a horizontal bar chart showing class probabilities.

        Args:
            probabilities: Array of shape (num_classes,) with probabilities
            true_label: Ground truth class (1-indexed)
            predicted_label: Predicted class (1-indexed)
            height: Height of the output image
            width: Width of the output image

        Returns:
            RGB image of shape (height, width, 3)
        """
        num_classes = len(probabilities)

        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        # Class labels (1-indexed for display)
        class_labels = [f"Class {i+1}" for i in range(num_classes)]
        y_pos = np.arange(num_classes)

        # Create colors: green for true label, red for wrong prediction, blue for others
        colors = []
        for i in range(num_classes):
            class_id = i + 1  # 1-indexed
            if class_id == true_label and class_id == predicted_label:
                colors.append('#00ff88')  # Green - correct prediction
            elif class_id == true_label:
                colors.append('#ffaa00')  # Orange - true label (missed)
            elif class_id == predicted_label:
                colors.append('#ff4466')  # Red - wrong prediction
            else:
                colors.append('#4a90d9')  # Blue - other classes

        # Create horizontal bar chart
        bars = ax.barh(y_pos, probabilities * 100, color=colors, height=0.7, edgecolor='white', linewidth=0.5)

        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob > 0.02:  # Only show label if probability > 2%
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{prob*100:.1f}%', va='center', ha='left',
                       color='white', fontsize=7, fontweight='bold')

        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_labels, fontsize=8, color='white')
        ax.set_xlabel('Probability (%)', fontsize=9, color='white')
        ax.set_xlim(0, 105)
        ax.invert_yaxis()  # Highest probability at top
        ax.tick_params(axis='x', colors='white', labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        # Add legend
        legend_text = f"True: Class {true_label} | Pred: Class {predicted_label}"
        is_correct = true_label == predicted_label
        legend_color = '#00ff88' if is_correct else '#ff4466'
        ax.set_title(legend_text, fontsize=9, color=legend_color, pad=10, fontweight='bold')

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[:, :, :3]  # Remove alpha channel (RGBA -> RGB)

        plt.close(fig)

        # Resize to exact dimensions
        img = cv2.resize(img, (width, height))

        return img

    def create_visualization_with_chart(
        self,
        frames: np.ndarray,
        probabilities: np.ndarray,
        true_label: int,
        predicted_label: int,
        chart_width: int = 250,
        scale_factor: int = 3,
    ) -> np.ndarray:
        """
        Create visualization with video frames and probability chart side by side.

        Args:
            frames: Video frames of shape (T, H, W, 3)
            probabilities: Class probabilities from model output
            true_label: Ground truth class
            predicted_label: Predicted class
            chart_width: Width of the probability chart (before scaling)
            scale_factor: Factor to scale up frames and chart for better visibility

        Returns:
            Combined frames of shape (T, scaled_H, scaled_W + scaled_chart_width, 3)
        """
        num_frames, height, width, channels = frames.shape

        # Scale up dimensions for better text visibility
        scaled_height = height * scale_factor
        scaled_width = width * scale_factor
        scaled_chart_width = chart_width * scale_factor

        # Scale up frames
        scaled_frames = []
        for frame in frames:
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
            scaled_frames.append(scaled_frame)
        scaled_frames = np.array(scaled_frames)

        # Create the probability chart at scaled size
        chart = self.create_probability_chart(
            probabilities=probabilities,
            true_label=true_label,
            predicted_label=predicted_label,
            height=scaled_height,
            width=scaled_chart_width,
        )

        # Combine frames with chart
        combined_frames = []
        for frame in scaled_frames:
            combined = np.concatenate([frame, chart], axis=1)
            combined_frames.append(combined)

        return np.array(combined_frames)

    def create_visualization_with_dynamic_chart(
        self,
        frames: np.ndarray,
        probabilities_per_frame: List[np.ndarray],
        true_labels_per_frame: List[int],
        chart_width: int = 250,
        scale_factor: int = 3,
    ) -> np.ndarray:
        """
        Create visualization with video frames and dynamically updating probability chart.

        Args:
            frames: Video frames of shape (T, H, W, 3)
            probabilities_per_frame: List of probability arrays, one per frame
            true_labels_per_frame: List of true labels, one per frame
            chart_width: Width of the probability chart (before scaling)
            scale_factor: Factor to scale up frames and chart for better visibility

        Returns:
            Combined frames of shape (T, scaled_H, scaled_W + scaled_chart_width, 3)
        """
        num_frames, height, width, channels = frames.shape

        # Scale up dimensions for better text visibility
        scaled_height = height * scale_factor
        scaled_width = width * scale_factor
        scaled_chart_width = chart_width * scale_factor

        # Combine frames with dynamic chart
        combined_frames = []
        for i, frame in enumerate(frames):
            # Scale up frame
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

            # Get probabilities and label for this frame
            probs = probabilities_per_frame[i] if i < len(probabilities_per_frame) else probabilities_per_frame[-1]
            true_label = true_labels_per_frame[i] if i < len(true_labels_per_frame) else true_labels_per_frame[-1]
            predicted_label = int(np.argmax(probs))

            # Create chart for this frame
            chart = self.create_probability_chart(
                probabilities=probs,
                true_label=true_label,
                predicted_label=predicted_label,
                height=scaled_height,
                width=scaled_chart_width,
            )

            combined = np.concatenate([scaled_frame, chart], axis=1)
            combined_frames.append(combined)

        return np.array(combined_frames)

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
