#!/usr/bin/env python3
"""Quick test to generate real Sionna data"""

import sys
import json
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.multi_layer_generator import MultiLayerDataGenerator, DataGenerationConfig
from src.data_generation.features import RTFeatureExtractor, PHYFAPIFeatureExtractor, MACRRCFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load scene
    scene_path = Path("data/scenes/quick_test/scene_-105.275_40.016")
    metadata_path = scene_path / "metadata.json"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    logger.info(f"Scene metadata: {metadata['scene_id']}")
    logger.info(f"Extent: {metadata['bbox']}")
    
    # Extract site positions - sites is a list, position is already a list [x, y, z]
    sites = metadata['sites']
    site_positions = []
    site_metadata = []
    
    for site in sites:
        pos = site['position']  # Already [x, y, z]
        site_positions.append(pos)
        site_metadata.append({
            'cell_id': site['cell_id'],
            'sector_id': site['sector_id'],
            'power_dbm': site['power_dbm'],
            'orientation': site['antenna']['orientation']
        })
    
    logger.info(f"Loaded {len(site_positions)} sites")
    logger.info(f"First site position: {site_positions[0]}")
    
    # Create config with REAL Sionna enabled
    config = DataGenerationConfig(
        scene_dir=Path("data/scenes/quick_test"),
        scene_metadata_path=metadata_path,
        carrier_frequency_hz=3500000000,  # 3.5 GHz
        bandwidth_hz=100000000,  # 100 MHz
        use_mock_mode=False,  # ← REAL SIONNA!
        num_ue_per_tile=10,  # Small test
        max_depth=5,
        num_samples=100_000,  # 100K rays (fast)
        enable_diffraction=False,  # Disabled for 2x speed
    )
    
    logger.info(f"Config: use_mock_mode={config.use_mock_mode}")
    
    # Initialize generator
    gen = MultiLayerDataGenerator(config)
    
    # Initialize feature extractors (not actually needed, generator creates them internally)
    
    # Load Sionna scene
    logger.info("Loading Sionna scene...")
    scene = gen._load_sionna_scene(scene_path / "scene.xml")
    logger.info(f"Scene loaded: {scene}")
    
    # Setup transmitters
    logger.info("Setting up transmitters...")
    transmitters = gen._setup_transmitters(scene, site_positions, site_metadata)
    logger.info(f"Created {len(transmitters)} transmitters")
    
    # Generate a test UE position (center of scene)
    bbox = metadata['bbox']
    ue_x = (bbox[0] + bbox[2]) / 2
    ue_y = (bbox[1] + bbox[3]) / 2
    ue_position = [ue_x * 111000, ue_y * 111000, 1.5]  # Convert to meters, 1.5m height
    
    logger.info(f"UE position: {ue_position}")
    
    # Get cell IDs
    cell_ids = [s['cell_id'] for s in site_metadata]
    
    # Run ray tracing!
    logger.info("Running Sionna ray tracing...")
    rt_feat, phy_feat, mac_feat = gen._simulate_rf(
        ue_position=ue_position,
        site_positions=site_positions,
        cell_ids=cell_ids,
        scene=scene
    )
    
    logger.info("SUCCESS! Real Sionna data generated")
    logger.info(f"RT features: {rt_feat}")
    logger.info(f"Path gains shape: {rt_feat.path_gains.shape}")
    logger.info(f"Path gains std: {np.std(rt_feat.path_gains)}")
    logger.info(f"RSRP mean: {np.mean(phy_feat.rsrp_dbm)}")
    logger.info(f"RSRP std: {np.std(phy_feat.rsrp_dbm)}")

if __name__ == "__main__":
    main()
