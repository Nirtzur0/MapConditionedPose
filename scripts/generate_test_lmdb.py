#!/usr/bin/env python3
"""Generate a small LMDB dataset for testing."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/CellularPositioningResearch')

from pathlib import Path
from src.data_generation.multi_layer_generator import MultiLayerDataGenerator
from src.data_generation.config import DataGenerationConfig

# Use existing scenes
scene_dir = Path('/home/ubuntu/projects/CellularPositioningResearch/outputs/quick_test/scenes')

# Create config for test dataset
config = DataGenerationConfig(
    scene_dir=scene_dir,
    scene_metadata_path=scene_dir / 'metadata.json',
    carrier_frequency_hz=3.5e9,
    bandwidth_hz=100e6,
    tx_power_dbm=43.0,
    noise_figure_db=9.0,
    num_ue_per_tile=50,  # More samples
    num_reports_per_ue=10,
    output_dir=Path('/home/ubuntu/projects/CellularPositioningResearch/data/processed'),
    use_mock_mode=False,  # Use real Sionna
)

print("Generating LMDB dataset...")
generator = MultiLayerDataGenerator(config)

# Generate with splits (train/val/test)
# Since we only have 1 scene, we'll generate multiple batches
output_paths = generator.generate_dataset(
    create_splits=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print("\nDataset paths:")
if isinstance(output_paths, dict):
    for split, path in output_paths.items():
        print(f"  {split}: {path}")
else:
    print(f"  {output_paths}")

print("\n✓ LMDB dataset generation complete!")
