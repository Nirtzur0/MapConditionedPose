#!/usr/bin/env python3
"""
Simplified End-to-End Pipeline
Direct function calls instead of subprocess wrappers

Usage:
    python run_pipeline_v2.py --config experiment.yaml
    python run_pipeline_v2.py --quick-test
"""

import argparse
import logging
import sys
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yaml

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ['matplotlib', 'PIL', 'tensorflow', 'numba']:
    logging.getLogger(name).setLevel(logging.WARNING)


@dataclass
class ExperimentConfig:
    """Unified configuration for the entire pipeline."""
    
    # Experiment metadata
    name: str = "experiment"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    clean: bool = False
    
    # Scene generation
    scenes: Dict = field(default_factory=lambda: {
        'cities': [
            {'name': 'Boulder, CO', 'bbox': [-105.28, 40.014, -105.27, 40.020]},
            {'name': 'Austin, TX', 'bbox': [-97.76, 30.275, -97.75, 30.285]},
            {'name': 'Seattle, WA', 'bbox': [-122.36, 47.625, -122.35, 47.635]},
            {'name': 'Chicago, IL', 'bbox': [-87.67, 41.875, -87.66, 41.885]},
            {'name': 'NYC, NY', 'bbox': [-73.985, 40.75, -73.975, 40.76]}
        ],
        'num_tx': 3,
        'site_strategy': 'random'
    })
    
    # Data generation
    data: Dict = field(default_factory=lambda: {
        'carrier_freq_hz': 3.5e9,
        'bandwidth_hz': 100e6,
        'num_ue_per_tile': 100,
        'num_reports_per_ue': 10,
        'split_ratios': {'train': 0.70, 'val': 0.15, 'test': 0.15}
    })
    
    # Training
    training: Dict = field(default_factory=lambda: {
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.0002,
        'num_workers': 4
    })
    
    # Pipeline control
    skip_scenes: bool = False
    skip_data: bool = False
    skip_training: bool = False
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def quick_test(cls) -> 'ExperimentConfig':
        """Create a quick test configuration."""
        return cls(
            name="quick_test",
            scenes={
                'cities': [{'name': 'Boulder, CO', 'bbox': [-105.275, 40.016, -105.272, 40.018]}],
                'num_tx': 2,
                'site_strategy': 'random'
            },
            data={
                'carrier_freq_hz': 3.5e9,
                'bandwidth_hz': 100e6,
                'num_ue_per_tile': 20,
                'num_reports_per_ue': 5,
                'split_ratios': {'train': 0.70, 'val': 0.15, 'test': 0.15}
            },
            training={
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 0.0002,
                'num_workers': 2
            }
        )


class Pipeline:
    """Simplified pipeline with direct function calls."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.project_root = Path(__file__).parent
        self.start_time = time.time()
        
        # Derived paths (convention over configuration)
        self.output_dir = self.project_root / "outputs" / config.name
        
        # Use existing scenes if available, otherwise create new
        existing_scenes = self.project_root / "data" / "scenes"
        if config.skip_scenes and existing_scenes.exists():
            self.scene_dir = existing_scenes
        else:
            self.scene_dir = self.output_dir / "scenes"
            
        self.data_dir = self.output_dir / "data"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
        # Dataset paths (set during data generation)
        self.train_path: Optional[Path] = None
        self.val_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.yaml", 'w') as f:
            yaml.dump({
                'name': config.name,
                'scenes': config.scenes,
                'data': config.data,
                'training': config.training
            }, f, default_flow_style=False)
    
    def log(self, msg: str, level: str = "info"):
        """Simple logging."""
        getattr(logger, level)(msg)
    
    def run(self) -> int:
        """Execute the pipeline."""
        try:
            self.log(f"Starting pipeline: {self.config.name}")
            self.log(f"Output directory: {self.output_dir}")
            
            if not self.config.skip_scenes:
                self.generate_scenes()
            
            if not self.config.skip_data:
                self.generate_data()
            
            if not self.config.skip_training:
                self.train_model()
            
            self.generate_report()
            
            duration = time.time() - self.start_time
            self.log(f"✓ Pipeline completed in {duration/60:.1f} minutes")
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return 1
    
    def generate_scenes(self):
        """Generate 3D scenes with transmitter sites."""
        self.log("=" * 60)
        self.log("STEP 1: Generate Scenes")
        self.log("=" * 60)
        
        from src.scene_generation import SceneGenerator, SitePlacer
        
        scene_config = self.config.scenes
        cities = scene_config.get('cities', [])
        
        for city in cities:
            city_name = city.get('name', 'unknown')
            bbox = city.get('bbox')
            
            self.log(f"Generating scene for: {city_name}")
            
            # Create scene directory
            slug = city_name.lower().replace(', ', '_').replace(' ', '_')
            city_scene_dir = self.scene_dir / slug
            city_scene_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize generator
            generator = SceneGenerator(
                scene_builder_path=str(self.project_root / "src" / "scene_builder"),
                site_placer=SitePlacer(strategy=scene_config.get('site_strategy', 'random')),
                output_dir=city_scene_dir
            )
            
            # Convert bbox to polygon
            if bbox:
                west, south, east, north = bbox
                polygon = [(west, south), (east, south), (east, north), (west, north), (west, south)]
            else:
                raise ValueError(f"No bbox specified for {city_name}")
            
            # Generate scene
            result = generator.generate(
                polygon_points=polygon,
                scene_id=f"scene_{slug}",
                site_config={'num_tx': scene_config.get('num_tx', 3)}
            )
            
            self.log(f"✓ Generated: {result.get('scene_path', city_scene_dir)}")
    
    def generate_data(self):
        """Generate training data using Sionna ray tracing."""
        self.log("=" * 60)
        self.log("STEP 2: Generate Dataset")
        self.log("=" * 60)
        
        from src.data_generation.multi_layer_generator import MultiLayerDataGenerator
        from src.data_generation.config import DataGenerationConfig
        
        data_config = self.config.data
        
        # Create data generation config
        config = DataGenerationConfig(
            scene_dir=self.scene_dir,
            scene_metadata_path=self.scene_dir / 'metadata.json',
            carrier_frequency_hz=data_config.get('carrier_freq_hz', 3.5e9),
            bandwidth_hz=data_config.get('bandwidth_hz', 100e6),
            tx_power_dbm=data_config.get('tx_power_dbm', 43.0),
            noise_figure_db=data_config.get('noise_figure_db', 9.0),
            num_ue_per_tile=data_config.get('num_ue_per_tile', 100),
            num_reports_per_ue=data_config.get('num_reports_per_ue', 10),
            output_dir=self.data_dir,
        )
        
        # Generate dataset
        generator = MultiLayerDataGenerator(config)
        
        split_ratios = data_config.get('split_ratios', {'train': 0.70, 'val': 0.15, 'test': 0.15})
        
        output_paths = generator.generate_dataset(
            create_splits=True,
            train_ratio=split_ratios['train'],
            val_ratio=split_ratios['val'],
            test_ratio=split_ratios['test']
        )
        
        # Store paths
        if isinstance(output_paths, dict):
            self.train_path = output_paths.get('train')
            self.val_path = output_paths.get('val')
            self.test_path = output_paths.get('test')
            
            self.log(f"✓ Train: {self.train_path}")
            self.log(f"✓ Val: {self.val_path}")
            self.log(f"✓ Test: {self.test_path}")
        else:
            self.log(f"✓ Dataset: {output_paths}")
    
    def train_model(self):
        """Train the transformer model."""
        self.log("=" * 60)
        self.log("STEP 3: Train Model")
        self.log("=" * 60)
        
        if not self.train_path or not self.val_path:
            # Try to find existing datasets
            self.train_path = self._find_latest_dataset('train')
            self.val_path = self._find_latest_dataset('val')
            self.test_path = self._find_latest_dataset('test')
        
        if not self.train_path:
            raise RuntimeError("No training dataset found. Run data generation first.")
        
        # Load base model config and merge with pipeline settings
        model_config_path = Path("configs/model.yaml")
        if model_config_path.exists():
            with open(model_config_path) as f:
                base_config = yaml.safe_load(f)
        else:
            # Fallback default config
            base_config = {}
        
        # Create training config by merging base config with pipeline settings
        training_config = {
            'dataset': {
                'train_zarr_paths': [str(self.train_path)],
                'val_zarr_paths': [str(self.val_path)] if self.val_path else [],
                'test_zarr_paths': [str(self.test_path)] if self.test_path else [],
                'map_resolution': 1.0,
                'scene_extent': 512,
                'normalize_features': True,
                'handle_missing_values': 'mask'
            },
            'training': {
                **base_config.get('training', {}),
                **self.config.training,
            },
            'infrastructure': {
                'accelerator': 'auto',
                'devices': 1,
                'num_workers': self.config.training.get('num_workers', 4),
                'precision': '32-true',
                'checkpoint': {
                    'dirpath': str(self.checkpoint_dir),
                    'monitor': 'val_median_error',
                    'mode': 'min',
                    'save_top_k': 3
                },
                'early_stopping': {
                    'monitor': 'val_median_error',
                    'patience': 10,
                    'mode': 'min'
                },
                'logging': {
                    'use_comet': bool(os.environ.get('COMET_API_KEY')),
                    'use_wandb': False,
                    'project': 'ue-localization',
                    'log_every_n_steps': 50
                }
            },
            # Use full model config from base
            'model': base_config.get('model', {
                'name': 'MapConditionedTransformer',
                'radio_encoder': {'type': 'SetTransformer', 'd_model': 256, 'nhead': 8, 'num_layers': 4, 'num_cells': 512, 'num_beams': 64},
                'map_encoder': {'d_model': 384, 'nhead': 6, 'num_layers': 6, 'patch_size': 16},
                'fusion': {'d_fusion': 384, 'nhead': 6},
                'coarse_head': {'grid_size': 32, 'd_input': 384},
                'fine_head': {'type': 'heteroscedastic', 'd_input': 512, 'd_hidden': 256, 'top_k': 5, 'patch_size': 64}
            }),
            'seed': 42,
            'deterministic': False
        }
        
        # Save training config
        config_path = self.checkpoint_dir / "training_config.yaml"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        # Train using Lightning
        from src.training import UELocalizationLightning
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        model = UELocalizationLightning(str(config_path))
        
        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename='model-{epoch:02d}-{val_median_error:.2f}',
                monitor='val_median_error',
                mode='min',
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor='val_median_error',
                patience=10,
                mode='min'
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=self.config.training.get('epochs', 30),
            accelerator='auto',
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=50
        )
        
        self.log(f"Training for {self.config.training.get('epochs', 30)} epochs...")
        trainer.fit(model)
        
        if self.test_path:
            self.log("Running test evaluation...")
            trainer.test(model)
        
        self.log(f"✓ Training complete. Checkpoints: {self.checkpoint_dir}")
    
    def _find_latest_dataset(self, split: str) -> Optional[Path]:
        """Find the latest dataset for a given split."""
        pattern = f"dataset_{split}_*.zarr"
        matches = sorted(self.data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None
    
    def generate_report(self):
        """Generate pipeline execution report."""
        duration = time.time() - self.start_time
        
        report = {
            'name': self.config.name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'duration_formatted': f"{duration/60:.1f} minutes",
            'output_dir': str(self.output_dir),
            'datasets': {
                'train': str(self.train_path) if self.train_path else None,
                'val': str(self.val_path) if self.val_path else None,
                'test': str(self.test_path) if self.test_path else None
            },
            'checkpoints': str(self.checkpoint_dir)
        }
        
        report_path = self.output_dir / "report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        self.log(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Simplified ML Pipeline")
    parser.add_argument('--config', type=Path, help='Path to experiment YAML config')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test')
    parser.add_argument('--skip-scenes', action='store_true', help='Skip scene generation')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    # Create config
    if args.quick_test:
        config = ExperimentConfig.quick_test()
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    # Apply overrides
    if args.name:
        config.name = args.name
    config.skip_scenes = args.skip_scenes
    config.skip_data = args.skip_data
    config.skip_training = args.skip_training
    
    # Run pipeline
    pipeline = Pipeline(config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    main()
