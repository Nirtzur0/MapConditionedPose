
import logging
import sys
from pathlib import Path
import numpy as np
import shutil

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data_generation.multi_layer_generator import MultiLayerDataGenerator, DataGenerationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_fix():
    print("Testing coordinate normalization fix...")
    
    # Define paths
    scene_dir = Path("data/scenes/austin_texas")
    output_dir = Path("data/processed/test_fix_output")
    
    # specific scene
    scene_id = "scene_-97.765_30.27"
    
    # Create simple config
    config = DataGenerationConfig(
        scene_dir=scene_dir,
        scene_metadata_path=scene_dir / scene_id / "metadata.json", # Pointing directly to scene metadata might be tricky if code expects root metadata
        # Wait, MultiLayerDataGenerator._load_scene_metadata expects {scene_dir}/{scene_id}/metadata.json
        # So scene_dir should be data/scenes/austin_texas
        
        output_dir=output_dir,
        num_ue_per_tile=1,      # Minimal UEs
        num_reports_per_ue=1,   # Minimal reports
        zarr_chunk_size=1,
        
        # Override critical params
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=100e6,
        tx_power_dbm=43.0,
        noise_figure_db=9.0,
    )
    
    # Initialize generator
    generator = MultiLayerDataGenerator(config)
    
    # Run generation for single scene
    print(f"Generating data for {scene_id}...")
    try:
        # We need to explicitly point to the scene metadata if the standard loading fails
        # But _load_scene_metadata look at config.scene_dir / scene_id / metadata.json
        # Check if that exists
        meta_path = scene_dir / scene_id / "metadata.json"
        if not meta_path.exists():
            print(f"Error: {meta_path} does not exist!")
            return

        output_path = generator.generate_dataset(scene_ids=[scene_id])
        print(f"Generation complete: {output_path}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix()
