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

    # Train on multiple locations, evaluate on another
    python run_pipeline.py --train-datasets data/processed/a.zarr data/processed/b.zarr --eval-dataset data/processed/c.zarr
"""

import argparse
import logging
import subprocess
import sys
import time
import os
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json
import shutil
import yaml
from easydict import EasyDict

from src.utils.logging_utils import setup_logging
from src.training import UELocalizationLightning
from src.training.optimization import run_optimization
from src.pipeline.scene_generation import generate_scenes
from src.pipeline.data_generation import generate_dataset
from src.pipeline.training import train_model, _apply_optuna_params, _create_training_config
from src.pipeline.evaluation import evaluate_model

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
        self.train_dataset_paths = [Path(p) for p in args.train_datasets] if args.train_datasets else []
        self.eval_dataset_path = Path(args.eval_dataset) if args.eval_dataset else None
        self.eval_config_path = None
        self.optuna_params: Optional[Dict[str, float]] = None
        self.optuna_config_path: Optional[Path] = None
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_data_output_path(self, config_path: Path) -> Optional[Path]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        data_gen = config.get('data_generation', {})
        output_path = data_gen.get('output', {}).get('path') or config.get('output', {}).get('path')
        output_dir = config.get('output_dir')
        if output_path:
            full_path = self.project_root / output_path
            if full_path.is_dir():
                return self._latest_dataset_in_dir(full_path)
            else:
                return full_path
        if output_dir:
            return self._latest_dataset_in_dir(self.project_root / output_dir)
        return None

    def _latest_dataset_in_dir(self, output_dir: Path) -> Optional[Path]:
        if not output_dir.exists():
            return None
        zarr_files = sorted(output_dir.glob("dataset_*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
        return zarr_files[0] if zarr_files else None

    def _run_dataset_generation_for_config(self, config_path: Path):
        cmd = [sys.executable, "scripts/generate_dataset.py", "--config", str(config_path)]
        dataset_dir = None
        scene_dir_from_config = None
        raw_config = None
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            if 'data_generation' in raw_config:
                data_gen = EasyDict(raw_config).data_generation
                scene_dir_from_config = self.project_root / data_gen.scenes.root_dir
                dataset_dir = self.project_root / Path(data_gen.output.path).parent
                cmd.extend(["--scene-dir", str(scene_dir_from_config)])
                cmd.extend(["--output-dir", str(dataset_dir)])
            else:
                scene_dir_from_config = self.project_root / raw_config.get('scene_dir')
                dataset_dir = self.project_root / raw_config.get('output_dir')
                if scene_dir_from_config:
                    cmd.extend(["--scene-dir", str(scene_dir_from_config)])
                if dataset_dir:
                    cmd.extend(["--output-dir", str(dataset_dir)])
        except Exception:
            pass

        if dataset_dir and self.args.clean and dataset_dir.exists():
            logger.info(f"Cleaning existing dataset at {dataset_dir}")
            shutil.rmtree(dataset_dir)
        if dataset_dir:
            dataset_dir.mkdir(parents=True, exist_ok=True)

        self.run_command(cmd, "Dataset Generation")

        self._maybe_generate_radio_maps(raw_config, scene_dir_from_config)

    def _maybe_generate_radio_maps(self, raw_config: Optional[Dict], scene_dir: Optional[Path]):
        if not raw_config or not scene_dir:
            return

        debug_flag = False
        output_dir = None
        if 'data_generation' in raw_config:
            debug_flag = raw_config['data_generation'].get('debug_radio_maps', False)
            output_dir = raw_config['data_generation'].get('radio_maps_output_dir')
        else:
            debug_flag = raw_config.get('debug_radio_maps', False)
            output_dir = raw_config.get('radio_maps_output_dir')

        if not debug_flag:
            return

        output_dir = output_dir or "data/radio_maps_debug"
        plots_dir = Path(output_dir) / "plots"

        cmd = [
            sys.executable,
            "scripts/generate_radio_maps.py",
            "--scenes-dir",
            str(scene_dir),
            "--output-dir",
            str(self.project_root / output_dir),
            "--pattern",
            "**/scene_*/scene.xml",
            "--save-plots",
            "--plots-dir",
            str(self.project_root / plots_dir),
        ]
        self.run_command(cmd, "Radio Map Debug Plots")
        
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
        generate_scenes(self.args, self.project_root, self.scene_dir, self.log_section, self.run_command)
        
    def step_2_generate_dataset(self):
        """Generate synthetic dataset from scenes using Sionna ray tracing"""
        self.dataset_path, self.eval_dataset_path = generate_dataset(
            self.args, self.project_root, self.scene_dir, self.dataset_dir,
            self.train_dataset_paths, self.eval_dataset_path,
            self.log_section, self.run_command
        )
        
    def step_3_train_model(self):
        """Train the transformer model"""
        train_model(self.args, self.project_root, self.checkpoint_dir, self.optuna_config_path,
                   self.optuna_params, self.train_dataset_paths, self.dataset_path, self.eval_dataset_path,
                   self.args.num_tx, self.log_section, self.run_command)

    def step_3_optimize_model(self):
        """Run Optuna hyperparameter optimization."""
        if not self.args.optimize:
            return

        self.log_section("STEP 3: Optimize Model (Optuna)")

        dataset_paths = self.train_dataset_paths or ([self.dataset_path] if self.dataset_path else [])
        base_config_path = _create_training_config(self.args, self.project_root, self.checkpoint_dir, dataset_paths, self.eval_dataset_path, self.args.num_tx)
        self.optuna_params = run_optimization(self.args, base_config_path)

        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = _apply_optuna_params(config, self.optuna_params)

        self.optuna_config_path = self.checkpoint_dir / "training_config_optuna.yaml"
        with open(self.optuna_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Optuna best config saved to: {self.optuna_config_path}")

    def step_4_eval_model(self):
        """Evaluate trained model on held-out dataset."""
        evaluate_model(self.args, self.checkpoint_dir, self.eval_dataset_path,
                      self.log_section, self.run_command)
        
    def step_5_generate_report(self):
        """Generate pipeline execution report"""
        step_label = "STEP 6: Generate Report" if self.args.optimize else "STEP 5: Generate Report"
        self.log_section(step_label)
        
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
                "train_datasets": [str(p) for p in self.train_dataset_paths] if self.train_dataset_paths else None,
                "eval_dataset": str(self.eval_dataset_path) if self.eval_dataset_path else None,
            },
            "steps_completed": []
        }
        
        if not self.args.skip_scenes:
            report["steps_completed"].append("Scene Generation")
        if not self.args.skip_dataset:
            report["steps_completed"].append("Dataset Generation")
        if self.args.optimize:
            report["steps_completed"].append("Hyperparameter Optimization")
        if not self.args.skip_training:
            report["steps_completed"].append("Model Training")
        if self.eval_dataset_path is not None and not self.args.skip_eval:
            report["steps_completed"].append("Model Evaluation")
            
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
            
            if self.args.optimize:
                self.step_3_optimize_model()
            
            if not self.args.skip_training:
                self.step_3_train_model()
                self.step_4_eval_model()
            elif self.eval_dataset_path is not None:
                self.step_4_eval_model()
                
            self.step_5_generate_report()
            
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
    parser.add_argument('--scene-config', type=Path, default=None,
                       help='Path to scene generation YAML config file')
    parser.add_argument('--data-config', type=Path, default=None,
                       help='Path to data generation YAML config file')
    parser.add_argument('--train-data-configs', type=Path, nargs='+', default=None,
                       help='Data generation configs for training (multi-location)')
    parser.add_argument('--eval-data-config', type=Path, default=None,
                       help='Data generation config for evaluation-only location')
    parser.add_argument('--skip-scenes', action='store_true',
                       help='Skip scene generation')
    parser.add_argument('--skip-dataset', action='store_true',
                       help='Skip dataset generation')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (skip scenes and dataset)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (skip scenes, dataset, training)')
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
    parser.add_argument('--train-datasets', nargs='+', default=None,
                       help='Paths to training Zarr datasets (skip generation)')
    parser.add_argument('--eval-dataset', type=Path, default=None,
                       help='Path to evaluation Zarr dataset (skip generation)')
    
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
    parser.add_argument('--comet-api-key', type=str, default='1lc3SG8vCkNzrn5p9ZmZs328K',
                       help='Comet ML API key (or set COMET_API_KEY env var)')
    parser.add_argument('--comet-workspace', type=str, default='nirtzur0',
                       help='Comet ML workspace (or set COMET_WORKSPACE env var)')
    parser.add_argument('--comet-project', type=str, default='ue-localization',
                       help='Comet ML project name (or set COMET_PROJECT_NAME env var)')
    parser.add_argument('--run-name', type=str,
                       default=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Run name for tracking')
    parser.add_argument('--light-train', action='store_true',
                       help='Use light training defaults for quick checks')

    # Optimization parameters
    parser.add_argument('--optimize', action='store_true',
                          help='Enable hyperparameter optimization with Optuna')
    parser.add_argument('--n-trials', type=int, default=50,
                            help='Number of optimization trials to run')
    parser.add_argument('--study-name', type=str, default='ue-localization-optimization',
                            help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                            help='Database URL for Optuna study storage')
    
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

    if args.light_train:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 8)
        logger.info("Light training mode enabled")
    
    # Handle train-only mode
    if args.train_only:
        args.skip_scenes = True
        args.skip_dataset = True
        args.skip_eval = True

    if args.eval_only:
        args.skip_scenes = True
        args.skip_dataset = True
        args.skip_training = True
        if not (args.eval_dataset or args.eval_data_config):
            raise ValueError("--eval-only requires --eval-dataset or --eval-data-config")
    
    return args


def _setup_comet_environment(args):
    """Set Comet ML environment variables from command-line arguments.
    
    Priority order:
    1. Existing environment variables (from shell scripts like run_full_experiment.sh)
    2. Command-line arguments / defaults
    """
    if args.comet:
        # Only set from args if env var is not already set (shell script takes precedence)
        if not os.environ.get('COMET_API_KEY') and args.comet_api_key:
            os.environ['COMET_API_KEY'] = args.comet_api_key
        if not os.environ.get('COMET_WORKSPACE') and args.comet_workspace:
            os.environ['COMET_WORKSPACE'] = args.comet_workspace
        if not os.environ.get('COMET_PROJECT_NAME') and args.comet_project:
            os.environ['COMET_PROJECT_NAME'] = args.comet_project
        
        # Check if API key is available
        if not os.environ.get('COMET_API_KEY'):
            logger.warning("⚠️  --comet enabled but no API key found.")
            logger.warning("   Set via --comet-api-key or COMET_API_KEY env var.")
        else:
            logger.info(f"✓ Comet ML enabled (workspace: {os.environ.get('COMET_WORKSPACE', 'default')})")


def main():
    args = parse_args()
    setup_logging(name=args.run_name)
    
    # Setup Comet ML environment before running pipeline
    _setup_comet_environment(args)

    orchestrator = PipelineOrchestrator(args)
    sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
