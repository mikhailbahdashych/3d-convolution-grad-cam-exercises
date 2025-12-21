"""3D CNN architecture for video classification."""

import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    C3D-style 3D Convolutional Network for video classification.

    Architecture:
        - 5 conv blocks with 3x3x3 kernels
        - BatchNorm + ReLU + MaxPool3D
        - Global Average Pooling
        - Fully connected classifier
    """

    def __init__(self, num_classes: int = 17, dropout: float = 0.5):
        """
        Initialize C3D model.

        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        # Block 1: (3, 16, 112, 112) -> (64, 16, 56, 56)
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # Block 2: (64, 16, 56, 56) -> (128, 8, 28, 28)
        self.block2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 3: (128, 8, 28, 28) -> (256, 4, 14, 14)
        self.block3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 4: (256, 4, 14, 14) -> (512, 2, 7, 7)
        self.block4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Block 5: (512, 2, 7, 7) -> (512, 1, 3, 3)
        self.block5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Global Average Pooling: (512, 1, 3, 3) -> (512,)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),  # Slightly lower dropout
            nn.Linear(256, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        # Conv blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, 512)

        # Classifier
        x = self.classifier(x)

        return x

    def get_feature_maps(self, x: torch.Tensor, block_name: str = "block5") -> torch.Tensor:
        """
        Get intermediate feature maps for GradCAM.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
            block_name: Name of block to extract features from

        Returns:
            Feature maps from specified block
        """
        # Forward pass up to specified block
        x = self.block1(x)
        if block_name == "block1":
            return x

        x = self.block2(x)
        if block_name == "block2":
            return x

        x = self.block3(x)
        if block_name == "block3":
            return x

        x = self.block4(x)
        if block_name == "block4":
            return x

        x = self.block5(x)
        return x


def build_c3d(config) -> C3D:
    """
    Build C3D model from config.

    Args:
        config: Configuration object

    Returns:
        C3D model
    """
    return C3D(
        num_classes=config.get("model.num_classes", 17),
        dropout=config.get("model.dropout", 0.5),
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = C3D(num_classes=17, dropout=0.5)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test feature extraction
    features = model.get_feature_maps(dummy_input, "block5")
    print(f"Block5 features shape: {features.shape}")
