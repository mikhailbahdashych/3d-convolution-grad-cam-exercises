"""Test training initialization."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.utils.device import get_device
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_train_transforms
from src.models.cnn3d import build_c3d, count_parameters
from src.training.losses import build_loss_fn
import torch

def main():
    print("Testing training initialization...\n")

    # Load config
    config = load_config()
    print("✓ Config loaded")

    # Get device
    device = get_device()
    print(f"✓ Device: {device}")

    # Create small dataset (just 10 samples for testing)
    transform = get_train_transforms(config)
    dataset = ExerciseVideoDataset(
        data_root=config.get("data.root_dir"),
        split="train",
        clip_length=config.get("preprocessing.clip_length"),
        temporal_stride=config.get("preprocessing.temporal_stride"),
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=transform,
        filter_background=True,
    )
    print(f"✓ Dataset created: {len(dataset)} clips")

    # Test loading one sample
    clip, label, metadata = dataset[0]
    print(f"✓ Sample loaded: clip shape={clip.shape}, label={label}")

    # Create model
    model = build_c3d(config)
    print(f"✓ Model created: {count_parameters(model):,} parameters")

    # Test forward pass
    model = model.to(device)
    clip_batch = clip.unsqueeze(0).to(device)
    output = model(clip_batch)
    print(f"✓ Forward pass: output shape={output.shape}")

    # Create loss function
    class_weights = dataset.get_class_weights().to(device)
    criterion = build_loss_fn(config, class_weights)
    print(f"✓ Loss function: {type(criterion).__name__}")

    # Test loss computation
    label_tensor = torch.tensor([label]).to(device)
    loss = criterion(output, label_tensor)
    print(f"✓ Loss computed: {loss.item():.4f}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"✓ Optimizer created")

    # Test backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"✓ Backward pass successful")

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nYou can now start training with:")
    print("  python scripts/train.py --epochs 2")

if __name__ == "__main__":
    main()
