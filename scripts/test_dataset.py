"""Test dataset loading."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import load_config
from src.data.dataset import ExerciseVideoDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

def main():
    config = load_config()

    print("Testing dataset loading...")

    # Create dataset
    train_transform = get_train_transforms(config)
    train_dataset = ExerciseVideoDataset(
        data_root=config.get("data.root_dir"),
        split="train",
        clip_length=config.get("preprocessing.clip_length"),
        temporal_stride=config.get("preprocessing.temporal_stride"),
        spatial_size=config.get("preprocessing.spatial_size"),
        transform=train_transform,
        filter_background=True,
        cache_videos=False,
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")

    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(train_dataset))):
        clip, label, metadata = train_dataset[i]
        print(f"Sample {i}: clip shape={clip.shape}, label={label}, subject={metadata['subject_id']}")

    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    for batch_idx, (clips, labels, metadata) in enumerate(train_loader):
        print(f"Batch {batch_idx}: clips shape={clips.shape}, labels shape={labels.shape}")
        if batch_idx >= 2:
            break

    print("\n+ Dataset test passed!")

if __name__ == "__main__":
    main()
