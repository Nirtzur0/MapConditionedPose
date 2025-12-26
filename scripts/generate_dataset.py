#!/usr/bin/env python3
"""
Example Script: Generate Multi-Layer Synthetic Dataset
Processes M1 scenes and generates training data for M3
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.multi_layer_generator import (
    MultiLayerDataGenerator, DataGenerationConfig
)

# Configure logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-layer synthetic dataset from M1 scenes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--scene-dir', type=Path, required=True,
                       help='Directory containing M1 scenes (scene_*/)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/synthetic'),
                       help='Output directory for Zarr dataset')
    parser.add_argument('--config', type=Path,
                       help='YAML configuration file (overrides CLI args)')
    
    # Scene selection
    parser.add_argument('--scene-ids', nargs='+',
                       help='Specific scene IDs to process (default: all)')
    parser.add_argument('--num-scenes', type=int,
                       help='Limit number of scenes (for testing)')
    
    # RF Parameters
    parser.add_argument('--carrier-freq', type=float, default=3.5e9,
                       help='Carrier frequency in Hz (default: 3.5 GHz)')
    parser.add_argument('--bandwidth', type=float, default=100e6,
                       help='System bandwidth in Hz (default: 100 MHz)')
    parser.add_argument('--tx-power', type=float, default=43.0,
                       help='Transmit power in dBm (default: 43 dBm)')
    parser.add_argument('--noise-figure', type=float, default=9.0,
                       help='Receiver noise figure in dB (default: 9 dB)')
    
    # Sampling Strategy
    parser.add_argument('--num-ue', type=int, default=100,
                       help='Number of UEs per tile')
    parser.add_argument('--num-reports', type=int, default=10,
                       help='Number of measurement reports per UE')
    parser.add_argument('--report-interval', type=float, default=200.0,
                       help='Time between reports in ms')
    parser.add_argument('--ue-height-min', type=float, default=1.5,
                       help='Min UE height in meters (pedestrian)')
    parser.add_argument('--ue-height-max', type=float, default=1.5,
                       help='Max UE height in meters')
    parser.add_argument('--ue-velocity-min', type=float, default=0.0,
                       help='Min UE velocity in m/s')
    parser.add_argument('--ue-velocity-max', type=float, default=1.5,
                       help='Max UE velocity in m/s (walking speed)')
    
    # Feature Extraction
    parser.add_argument('--enable-k-factor', action='store_true',
                       help='Compute Rician K-factor (slower)')
    parser.add_argument('--enable-beam-mgmt', action='store_true', default=True,
                       help='Enable 5G NR beam management (L1-RSRP per beam)')
    parser.add_argument('--num-beams', type=int, default=64,
                       help='Number of SSB beams (64 for mmWave, 8 for sub-6)')
    parser.add_argument('--max-neighbors', type=int, default=8,
                       help='Max neighbor cells to report (3GPP default: 8)')
    
    # Measurement Realism
    parser.add_argument('--disable-dropout', action='store_true',
                       help='Disable measurement dropout (perfect availability)')
    parser.add_argument('--disable-quantization', action='store_true',
                       help='Disable 3GPP quantization (floating point)')
    
    # Storage
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Zarr chunk size (samples per chunk)')
    parser.add_argument('--compression', choices=['blosc', 'gzip', 'lz4', 'none'],
                       default='blosc', help='Compression algorithm')
    parser.add_argument('--compression-level', type=int, default=5,
                       help='Compression level (1-9)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = DataGenerationConfig.from_yaml(args.config)
    else:
        # Build config from CLI args
        config = DataGenerationConfig(
            scene_dir=args.scene_dir,
            scene_metadata_path=args.scene_dir / 'metadata.json',
            carrier_frequency_hz=args.carrier_freq,
            bandwidth_hz=args.bandwidth,
            tx_power_dbm=args.tx_power,
            noise_figure_db=args.noise_figure,
            num_ue_per_tile=args.num_ue,
            ue_height_range=(args.ue_height_min, args.ue_height_max),
            ue_velocity_range=(args.ue_velocity_min, args.ue_velocity_max),
            num_reports_per_ue=args.num_reports,
            report_interval_ms=args.report_interval,
            enable_k_factor=args.enable_k_factor,
            enable_beam_management=args.enable_beam_mgmt,
            num_beams=args.num_beams,
            max_neighbors=args.max_neighbors,
            measurement_dropout_rates=None if args.disable_dropout else {
                'rsrp': 0.05, 'rsrq': 0.10, 'sinr': 0.10,
                'cqi': 0.15, 'ri': 0.20, 'pmi': 0.25,
                'neighbor_rsrp': 0.30,
            },
            quantization_enabled=not args.disable_quantization,
            output_dir=args.output_dir,
            zarr_chunk_size=args.chunk_size,
        )
    
    # Validate scene directory
    if not config.scene_dir.exists():
        logger.error(f"Scene directory not found: {config.scene_dir}")
        return 1
    
    # Check for scenes
    # Check for scenes (directories containing scene.xml)
    scene_xmls = list(config.scene_dir.rglob('scene.xml'))
    scene_dirs = [p.parent for p in scene_xmls]
    
    if len(scene_dirs) == 0:
        logger.error(f"No scenes found in {config.scene_dir}")
        logger.info("Expected to find 'scene.xml' files in subdirectories")
        return 1

    
    logger.info(f"Found {len(scene_dirs)} scenes in {config.scene_dir}")
    
    # Initialize generator
    logger.info("Initializing MultiLayerDataGenerator...")
    generator = MultiLayerDataGenerator(config)
    
    # Generate dataset
    logger.info("Starting dataset generation...")
    try:
        output_path = generator.generate_dataset(
            scene_ids=args.scene_ids,
            num_scenes=args.num_scenes
        )
        
        logger.info(f"✓ Dataset generation complete!")
        logger.info(f"  Output: {output_path}")
        
        # Print statistics
        stats = generator.zarr_writer.get_stats()
        logger.info(f"  Total samples: {stats['num_samples']}")
        logger.info(f"  Total scenes: {stats['num_scenes']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
