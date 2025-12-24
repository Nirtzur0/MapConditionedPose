#!/usr/bin/env python3
"""
End-to-End Pipeline Orchestrator
Runs the complete transformer-ue-localization pipeline from scene generation to training

Usage:
    # Quick test run
    python run_pipeline.py --quick-test
    
    # Full pipeline with custom parameters
    python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 3 --epochs 50
    
    # Resume from existing scene/dataset
    python run_pipeline.py --skip-scenes --skip-dataset --train-only
"""

import argparse
import logging
import subprocess
import sys
import time
import os
import re
from pathlib import Path
from datetime import datetime
import json
import shutil

from src.utils.logging_utils import setup_logging




logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete ML pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.start_time = time.time()
        
        # Pipeline paths
        self.scene_dir = self.project_root / "data" / "scenes" / args.scene_name
        self.dataset_dir = self.project_root / "data" / "processed" / f"{args.scene_name}_dataset"
        self.checkpoint_dir = self.project_root / "checkpoints" / args.run_name
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def log_section(self, title):
        """Log a section header"""
        logger.info("=" * 80)
        logger.info(f"  {title}")
        logger.info("=" * 80)
        
    def run_command(self, cmd, step_name, check=True):
        """Run a command and handle errors"""
        logger.info(f"Running: {' '.join(cmd)}")
        start = time.time()
        
        try:
            env = os.environ.copy()
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            process = subprocess.Popen(
                cmd,
                text=True,
                cwd=self.project_root,
                env=env
            )
            return_code = process.wait()
            if check and return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            duration = time.time() - start
            logger.info(f"✓ {step_name} completed in {duration:.1f}s")
            return return_code
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start
            logger.error(f"✗ {step_name} failed after {duration:.1f}s")
            raise
    
    def step_1_generate_scenes(self):
        """Generate 3D scenes with transmitter sites using GIS data"""
        if self.args.skip_scenes:
            logger.info("Skipping scene generation (--skip-scenes)")
            return
            
        self.log_section("STEP 1: Generate Scenes")
        logger.info(f"Using GIS bounding box: {self.args.bbox} for scene generation")
        
        # Clean existing scenes if requested
        if self.args.clean and self.scene_dir.exists():
            logger.info(f"Cleaning existing scenes at {self.scene_dir}")
            shutil.rmtree(self.scene_dir)
        
        cmd = [
            sys.executable,
            "scripts/generate_scenes.py",
            "--bbox",
            str(self.args.bbox[0]),  # west
            str(self.args.bbox[1]),  # south
            str(self.args.bbox[2]),  # east
            str(self.args.bbox[3]),  # north
            "--output", str(self.scene_dir),
            "--num-tx", str(self.args.num_tx),
            "--site-strategy", self.args.site_strategy,
        ]
        
        if self.args.tiles:
            cmd.append("--tiles")
            
        self.run_command(cmd, "Scene Generation")
        
        # Verify scenes were created
        if not self.scene_dir.exists():
            raise RuntimeError(f"Scene directory not created: {self.scene_dir}")
            
        scene_count = len(list(self.scene_dir.glob("scene_*")))
        logger.info(f"Created {scene_count} scene(s)")
        
    def step_2_generate_dataset(self):
        """Generate synthetic dataset from scenes using Sionna ray tracing"""
        if self.args.skip_dataset:
            logger.info("Skipping dataset generation (--skip-dataset)")
            zarr_files = sorted(self.dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not zarr_files:
                raise RuntimeError(f"No dataset available to reuse in: {self.dataset_dir}")
            self.dataset_path = zarr_files[0]
            return
            
        self.log_section("STEP 2: Generate Dataset")
        logger.info(f"Using scenes from {self.scene_dir} with Sionna ray tracing")
        
        # Clean existing dataset if requested
        if self.args.clean and self.dataset_dir.exists():
            logger.info(f"Cleaning existing dataset at {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
        
        cmd = [
            sys.executable,
            "scripts/generate_dataset.py",
            "--scene-dir", str(self.scene_dir),
            "--output-dir", str(self.dataset_dir),
            "--carrier-freq", str(self.args.carrier_freq),
            "--bandwidth", str(self.args.bandwidth),
            "--num-ue", str(self.args.num_ues),
            "--num-reports", str(self.args.num_trajectories // 10),  # Convert trajectories to reports
        ]
        
        if self.args.num_scenes:
            cmd.extend(["--num-scenes", str(self.args.num_scenes)])
            
        self.run_command(cmd, "Dataset Generation")
        
        # Verify dataset was created - use the most recent one
        zarr_files = sorted(self.dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zarr_files:
            raise RuntimeError(f"No dataset created in: {self.dataset_dir}")
            
        logger.info(f"Created dataset: {zarr_files[0].name}")
        self.dataset_path = zarr_files[0]
        
    def step_3_train_model(self):
        """Train the transformer model"""
        if self.args.skip_training:
            logger.info("Skipping training (--skip-training)")
            return
            
        self.log_section("STEP 3: Train Model")
        
        # Use existing config or create custom one
        if self.args.config:
            config_path = self.args.config
        else:
            config_path = self._create_training_config()
        
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", str(config_path),
        ]
        
        if self.args.resume_checkpoint:
            cmd.extend(["--resume", str(self.args.resume_checkpoint)])
            
        if self.args.wandb:
            cmd.extend(["--wandb-project", "transformer-ue-localization"])
            cmd.extend(["--run-name", self.args.run_name])
        if self.args.comet:
            cmd.extend(["--run-name", self.args.run_name])
            
        self.run_command(cmd, "Model Training")
        
        logger.info(f"Training complete. Checkpoints saved to {self.checkpoint_dir}")
        
    def _create_training_config(self):
        """Create a custom training config based on pipeline parameters"""
        config_path = self.checkpoint_dir / "training_config.yaml"
        
        # Load base config
        base_config = self.project_root / "configs" / "training_simple.yaml"
        with open(base_config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Update dataset configuration with all required fields
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['zarr_path'] = str(self.dataset_path)
        config['dataset']['map_resolution'] = 2.0  # 2 meters per pixel
        config['dataset']['scene_extent'] = 856  # Scene size in meters
        config['dataset']['normalize_features'] = True
        config['dataset']['handle_missing_values'] = 'mask'
        config['metadata'] = {
            'run_name': self.args.run_name,
            'scene_name': self.args.scene_name,
            'dataset_path': str(self.dataset_path),
            'bbox': list(self.args.bbox),
            'num_tx': self.args.num_tx,
            'num_ues': self.args.num_ues,
            'num_trajectories': self.args.num_trajectories,
            'carrier_freq_hz': self.args.carrier_freq,
            'bandwidth_hz': self.args.bandwidth,
        }
        
        # Update training parameters
        if 'training' not in config:
            config['training'] = {}
        if 'data' not in config['training']:
            config['training']['data'] = {}
            
        config['training']['data']['batch_size'] = self.args.batch_size
        config['training']['data']['num_workers'] = 0  # Disable multiprocessing for Zarr v3 compatibility
        config['training']['epochs'] = self.args.epochs
        config['training']['num_epochs'] = self.args.epochs
        config['training']['learning_rate'] = self.args.learning_rate
        config['training']['batch_size'] = self.args.batch_size
        
        # Update infrastructure settings
        if 'infrastructure' not in config:
            config['infrastructure'] = {}
        config['infrastructure']['num_workers'] = 0  # Disable multiprocessing for Zarr v3 compatibility
        if 'logging' not in config['infrastructure']:
            config['infrastructure']['logging'] = {}
        config['infrastructure']['logging']['log_every_n_steps'] = 1
        
        # Update wandb settings
        if self.args.wandb:
            if 'wandb' not in config:
                config['wandb'] = {}
            config['wandb']['mode'] = 'online'
            if 'logging' not in config['infrastructure']:
                config['infrastructure']['logging'] = {}
            config['infrastructure']['logging']['use_wandb'] = True
        if self.args.comet:
            if 'logging' not in config['infrastructure']:
                config['infrastructure']['logging'] = {}
            config['infrastructure']['logging']['use_comet'] = True
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Created training config: {config_path}")
        return config_path
        
    def step_4_generate_report(self):
        """Generate pipeline execution report"""
        self.log_section("STEP 4: Generate Report")
        
        duration = time.time() - self.start_time
        
        report = {
            "pipeline_name": "transformer-ue-localization",
            "run_name": self.args.run_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "duration_formatted": f"{duration/60:.1f} minutes",
            "configuration": {
                "scene_bbox": self.args.bbox,
                "num_transmitters": self.args.num_tx,
                "num_ues": self.args.num_ues,
                "num_trajectories": self.args.num_trajectories,
                "carrier_freq_ghz": self.args.carrier_freq / 1e9,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
            },
            "outputs": {
                "scenes": str(self.scene_dir),
                "dataset": str(self.dataset_dir),
                "checkpoints": str(self.checkpoint_dir),
            },
            "steps_completed": []
        }
        
        if not self.args.skip_scenes:
            report["steps_completed"].append("Scene Generation")
        if not self.args.skip_dataset:
            report["steps_completed"].append("Dataset Generation")
        if not self.args.skip_training:
            report["steps_completed"].append("Model Training")
            
        report_path = self.checkpoint_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Pipeline report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("  PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Run Name:       {self.args.run_name}")
        logger.info(f"Duration:       {duration/60:.1f} minutes")
        logger.info(f"Steps:          {', '.join(report['steps_completed'])}")
        logger.info(f"Scenes:         {self.scene_dir}")
        logger.info(f"Dataset:        {self.dataset_dir}")
        logger.info(f"Checkpoints:    {self.checkpoint_dir}")
        logger.info("=" * 80 + "\n")
        
    def run(self):
        """Execute the complete pipeline"""
        try:
            logger.info(f"Starting pipeline: {self.args.run_name}")
            logger.info(f"Project root: {self.project_root}")
            
            self.step_1_generate_scenes()
            self.step_2_generate_dataset()
                
            if not self.args.skip_training:
                self.step_3_train_model()
                
            self.step_4_generate_report()
            
            logger.info("✓ Pipeline completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"✗ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (small area, few epochs)
  python run_pipeline.py --quick-test
  
  # Full pipeline
  python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
  
  # Skip scene generation (use existing)
  python run_pipeline.py --skip-scenes --scene-name boulder_test
  
  # Training only
  python run_pipeline.py --skip-scenes --skip-dataset --train-only
        """
    )
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test (small scene, few epochs)')
    
    # Pipeline control
    parser.add_argument('--skip-scenes', action='store_true',
                       help='Skip scene generation')
    parser.add_argument('--skip-dataset', action='store_true',
                       help='Skip dataset generation')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (skip scenes and dataset)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean existing outputs before running')
    
    # Scene parameters
    parser.add_argument('--bbox', type=float, nargs=4,
                       metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                       default=[-105.28, 40.014, -105.27, 40.020],
                       help='Bounding box coordinates')
    parser.add_argument('--scene-name', type=str, default='boulder_test',
                       help='Scene directory name')
    parser.add_argument('--num-tx', type=int, default=3,
                       help='Number of transmitters')
    parser.add_argument('--site-strategy', type=str, default='grid',
                       choices=['grid', 'random', 'cluster'],
                       help='Transmitter placement strategy')
    parser.add_argument('--tiles', action='store_true',
                       help='Generate multiple tiles')
    
    # Dataset parameters
    parser.add_argument('--carrier-freq', type=float, default=3.5e9,
                       help='Carrier frequency (Hz)')
    parser.add_argument('--bandwidth', type=float, default=100e6,
                       help='Bandwidth (Hz)')
    parser.add_argument('--num-ues', type=int, default=50,
                       help='Number of UEs per trajectory')
    parser.add_argument('--num-trajectories', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--num-scenes', type=int, default=None,
                       help='Limit number of scenes to process')
    
    # Training parameters
    parser.add_argument('--config', type=Path, default=None,
                       help='Training config file (auto-generated if not provided)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='DataLoader workers')
    parser.add_argument('--resume-checkpoint', type=Path, default=None,
                       help='Resume from checkpoint')
    
    # Logging
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--comet', action='store_true',
                       help='Enable Comet ML logging')
    parser.add_argument('--run-name', type=str,
                       default=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Run name for tracking')
    
    args = parser.parse_args()
    
    # Handle quick test mode
    if args.quick_test:
        args.bbox = [-105.275, 40.016, -105.272, 40.018]  # Tiny area
        args.num_tx = 2
        args.num_ues = 20
        args.num_trajectories = 50
        args.epochs = 5
        args.batch_size = 8
        args.scene_name = 'quick_test'
        args.run_name = 'quick_test'
        logger.info("Quick test mode enabled")
    
    # Handle train-only mode
    if args.train_only:
        args.skip_scenes = True
        args.skip_dataset = True
    
    return args


def main():
    args = parse_args()
    setup_logging(name=args.run_name)
    orchestrator = PipelineOrchestrator(args)
    sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
