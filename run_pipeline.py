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
from typing import Optional
import yaml
from omegaconf import OmegaConf

from src.config.schema import (
    load_pipeline_config,
    apply_quick_test_overrides,
    apply_robust_overrides,
)

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


class Pipeline:
    """Simplified pipeline with direct function calls."""
    
    def __init__(self, config):
        self.config = config
        self.project_root = Path(__file__).parent
        self.start_time = time.time()
        
        # Derived paths (convention over configuration)
        self.output_dir = Path(self.config.experiment.output_dir) / self.config.experiment.name
        
        # Use existing scenes if available, otherwise create new
        existing_scenes = self.project_root / "data" / "scenes"
        if self.config.pipeline.skip_scenes and existing_scenes.exists():
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
        self._save_config()

    def _save_config(self):
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(OmegaConf.to_container(self.config, resolve=True), f, sort_keys=False)
    
    def log(self, msg: str, level: str = "info"):
        """Simple logging."""
        getattr(logger, level)(msg)
    
    def run(self) -> int:
        """Execute the pipeline."""
        try:
            self.log(f"Starting pipeline: {self.config.experiment.name}")
            self.log(f"Output directory: {self.output_dir}")
            
            if not self.config.pipeline.skip_scenes:
                self.generate_scenes()
            
            if not self.config.pipeline.skip_data:
                self.generate_data()
            
            if not self.config.pipeline.skip_training:
                self.train_model()
            
            self.generate_report()
            
            duration = time.time() - self.start_time
            self.log(f"✓ Pipeline completed in {duration/60:.1f} minutes")
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return 1
    
    def generate_scenes(self):
        """Generate 3D scenes with transmitter sites.
        
        Supports multiple TX variations per city to improve data diversity.
        """
        self.log("=" * 60)
        self.log("STEP 1: Generate Scenes")
        self.log("=" * 60)
        
        from src.scene_generation import SceneGenerator, SitePlacer
        
        scene_config = self.config.scenes
        cities = scene_config.cities
        tx_variations = scene_config.tx_variations
        
        total_scenes = len(cities) * tx_variations
        self.log(f"Generating {len(cities)} cities × {tx_variations} TX variations = {total_scenes} scenes")
        
        for city in cities:
            city_name = city.name
            bbox = city.bbox
            split = city.split  # Track which split this belongs to
            
            # Generate multiple TX variations per city
            for var_idx in range(tx_variations):
                var_suffix = f"_v{var_idx}" if tx_variations > 1 else ""
                self.log(f"Generating scene for: {city_name} (variation {var_idx + 1}/{tx_variations})")
                
                # Create scene directory
                slug = city_name.lower().replace(', ', '_').replace(' ', '_')
                city_scene_dir = self.scene_dir / f"{slug}{var_suffix}"
                city_scene_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize generator with new random seed for each variation
                generator = SceneGenerator(
                    scene_builder_path=str(self.project_root / "src" / "scene_builder"),
                    site_placer=SitePlacer(
                        strategy=scene_config.site_strategy,
                        seed=hash(f"{city_name}_{var_idx}") % (2**32)  # Reproducible random seed
                    ),
                    output_dir=city_scene_dir
                )
                
                # Convert bbox to polygon
                if bbox:
                    west, south, east, north = bbox
                    polygon = [(west, south), (east, south), (east, north), (west, north), (west, south)]
                else:
                    raise ValueError(f"No bbox specified for {city_name}")
                
                # Generate scene
                scene_id = f"scene_{slug}{var_suffix}"
                result = generator.generate(
                    polygon_points=polygon,
                    scene_id=scene_id,
                    site_config={'num_tx': scene_config.num_tx}
                )
                
                # Store split metadata for later use in data generation
                metadata_path = city_scene_dir / scene_id / "split.txt"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, 'w') as f:
                    f.write(split)
                
                self.log(f"✓ Generated: {result.get('scene_path', city_scene_dir)} [{split}]")
    
    def generate_data(self):
        """Generate training data using Sionna ray tracing."""
        self.log("=" * 60)
        self.log("STEP 2: Generate Dataset")
        self.log("=" * 60)
        
        from src.data_generation.multi_layer_generator import MultiLayerDataGenerator
        from src.data_generation.config import DataGenerationConfig
        
        data_config = self.config.data_generation
        
        # Create data generation config
        config = DataGenerationConfig(
            scene_dir=self.scene_dir,
            scene_metadata_path=self.scene_dir / 'metadata.json',
            carrier_frequency_hz=data_config.carrier_frequency_hz,
            bandwidth_hz=data_config.bandwidth_hz,
            tx_power_dbm=data_config.tx_power_dbm,
            noise_figure_db=data_config.noise_figure_db,
            use_mock_mode=data_config.use_mock_mode,
            max_depth=data_config.max_depth,
            num_samples=data_config.num_samples,
            enable_diffraction=data_config.enable_diffraction,
            num_ue_per_tile=data_config.num_ue_per_tile,
            ue_height_range=tuple(data_config.ue_height_range),
            ue_velocity_range=tuple(data_config.ue_velocity_range),
            num_reports_per_ue=data_config.num_reports_per_ue,
            report_interval_ms=data_config.report_interval_ms,
            enable_k_factor=data_config.enable_k_factor,
            enable_beam_management=data_config.enable_beam_management,
            num_beams=data_config.num_beams,
            max_neighbors=data_config.max_neighbors,
            measurement_dropout_rates=data_config.measurement_dropout_rates,
            measurement_dropout_seed=data_config.measurement_dropout_seed,
            quantization_enabled=data_config.quantization_enabled,
            output_dir=self.data_dir,
            zarr_chunk_size=data_config.zarr_chunk_size,
            max_stored_paths=data_config.max_stored_paths,
            max_stored_sites=data_config.max_stored_sites,
        )
        
        # Generate dataset
        generator = MultiLayerDataGenerator(config)
        
        split_ratios = data_config.split_ratios
        
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
        
        if not self.train_path:
            # Try to find existing datasets
            self.train_path = self._find_latest_dataset('train')
            self.test_path = self._find_latest_dataset('test')
        
        if not self.train_path:
            raise RuntimeError("No training dataset found. Run data generation first.")
        
        # Try to find val dataset if not already set
        if not self.val_path:
            self.val_path = self._find_latest_dataset('val')
        
        # Log the dataset paths being used
        self.log(f"Using datasets:")
        self.log(f"  Train: {self.train_path}")
        self.log(f"  Val: {self.val_path if self.val_path else 'None (will use train for validation)'}")
        self.log(f"  Test: {self.test_path}")
        
        # Determine if using LMDB or Zarr based on file extension
        is_lmdb = str(self.train_path).endswith('.lmdb')

        dataset_cfg = self.config.dataset
        if is_lmdb:
            dataset_cfg.train_lmdb_paths = [str(self.train_path)]
            dataset_cfg.val_lmdb_paths = [str(self.val_path)] if self.val_path else []
            dataset_cfg.test_lmdb_paths = [str(self.test_path)] if self.test_path else []
        else:
            dataset_cfg.train_zarr_paths = [str(self.train_path)]
            dataset_cfg.val_zarr_paths = [str(self.val_path)] if self.val_path else []
            dataset_cfg.test_zarr_paths = [str(self.test_path)] if self.test_path else []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.infrastructure.checkpoint.dirpath = str(self.checkpoint_dir)

        # Save updated config (includes dataset paths)
        self._save_config()
        config_path = self.output_dir / "config.yaml"
        
        # Train using Lightning
        from src.training import UELocalizationLightning
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        
        model = UELocalizationLightning(str(config_path))
        
        checkpoint_cfg = self.config.infrastructure.checkpoint
        early_cfg = self.config.infrastructure.early_stopping

        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename='model-{epoch:02d}-{val_median_error:.2f}',
                monitor=checkpoint_cfg.monitor,
                mode=checkpoint_cfg.mode,
                save_top_k=checkpoint_cfg.save_top_k,
                save_last=True
            ),
            EarlyStopping(
                monitor=early_cfg.monitor,
                patience=early_cfg.patience,
                mode=early_cfg.mode
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=self.config.training.num_epochs,
            accelerator=self.config.infrastructure.accelerator,
            devices=self.config.infrastructure.devices,
            precision=self.config.infrastructure.precision,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=self.config.infrastructure.logging.log_every_n_steps
        )
        
        self.log(f"Training for {self.config.training.num_epochs} epochs...")
        trainer.fit(model)
        
        if self.test_path:
            self.log("Running test evaluation...")
            trainer.test(model)
        
        self.log(f"✓ Training complete. Checkpoints: {self.checkpoint_dir}")
    
    def _find_latest_dataset(self, split: str) -> Optional[Path]:
        """Find the latest LMDB dataset for a given split."""
        lmdb_pattern = f"dataset_*_{split}.lmdb"
        lmdb_matches = sorted(self.data_dir.glob(lmdb_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return lmdb_matches[0] if lmdb_matches else None
    
    def generate_report(self):
        """Generate pipeline execution report."""
        duration = time.time() - self.start_time
        
        report = {
            'name': self.config.experiment.name,
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
    parser.add_argument('--robust', action='store_true', help='Run robust training with extensive data diversity')
    parser.add_argument('--skip-scenes', action='store_true', help='Skip scene generation')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-training', action='store_true', help='Skip training')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
    args = parser.parse_args()
    
    default_config_path = Path("configs/pipeline.yaml")
    config_path = args.config or default_config_path
    config = load_pipeline_config(config_path)

    if args.quick_test:
        config = apply_quick_test_overrides(config)
    if args.robust:
        config = apply_robust_overrides(config)
    
    # Apply overrides
    if args.name:
        config.experiment.name = args.name
    config.pipeline.skip_scenes = args.skip_scenes
    config.pipeline.skip_data = args.skip_data
    config.pipeline.skip_training = args.skip_training
    
    # Run pipeline
    pipeline = Pipeline(config)
    sys.exit(pipeline.run())


if __name__ == "__main__":
    main()
