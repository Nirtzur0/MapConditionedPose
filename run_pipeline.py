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

# Import Optuna and Comet integration if available
try:
    import optuna
    from optuna_integration.comet import CometPruner, CometCallback
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

logger = logging.getLogger(__name__)


def objective(trial, args, base_config_path: Path):
    """
    Optuna objective function.
    
    This function is called for each trial in the study. It suggests
    hyperparameters, creates and trains a model, and returns the
    validation metric to be optimized.
    """
    print(f"Starting trial {trial.number}")

    # 1. Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Suggest hyperparameters
    # Radio Encoder
    config['model']['radio_encoder']['d_model'] = trial.suggest_categorical('radio_d_model', [256, 512])
    config['model']['radio_encoder']['nhead'] = trial.suggest_categorical('radio_nhead', [4, 8])
    config['model']['radio_encoder']['num_layers'] = trial.suggest_int('radio_num_layers', 2, 6)
    config['model']['radio_encoder']['dropout'] = trial.suggest_float('radio_dropout', 0.1, 0.3)

    # Map Encoder
    config['model']['map_encoder']['d_model'] = trial.suggest_categorical('map_d_model', [256, 512, 768])
    config['model']['map_encoder']['nhead'] = trial.suggest_categorical('map_nhead', [4, 8])
    config['model']['map_encoder']['num_layers'] = trial.suggest_int('map_num_layers', 2, 6)
    config['model']['map_encoder']['dropout'] = trial.suggest_float('map_dropout', 0.1, 0.3)

    # Fusion
    config['model']['fusion']['d_fusion'] = trial.suggest_categorical('fusion_d_fusion', [256, 512])
    config['model']['fusion']['nhead'] = trial.suggest_categorical('fusion_nhead', [4, 8])
    config['model']['fusion']['dropout'] = trial.suggest_float('fusion_dropout', 0.1, 0.3)
    
    # Training params
    config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])


    # 3. Create a temporary config file for the trial
    trial_config_path = Path(f"configs/trial_{trial.number}_config.yaml")
    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f)

    # 4. Create model
    model = UELocalizationLightning(str(trial_config_path))

    # 5. Setup callbacks
    callbacks = []
    
    # Optuna pruning callback
    try:
        from optuna_integration import PyTorchLightningPruningCallback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_median_error")
        callbacks.append(pruning_callback)
    except ImportError:
        pass

    # Checkpointing
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/trial_{trial.number}",
        filename='best_model',
        monitor=config['infrastructure']['checkpoint']['monitor'],
        mode=config['infrastructure']['checkpoint']['mode'],
        save_top_k=1,
    )
    callbacks.append(checkpoint_callback)
    
    # 6. Create trainer
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CometLogger

    comet_logger = None
    if COMET_AVAILABLE and os.environ.get('COMET_API_KEY'):
        comet_logger = CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            project_name=args.study_name,
            workspace=os.environ.get('COMET_WORKSPACE'),
        )
        comet_logger.experiment.set_name(f"trial_{trial.number}")
        comet_logger.experiment.log_parameters(trial.params)


    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['infrastructure']['accelerator'],
        devices=config['infrastructure']['devices'],
        precision=config['infrastructure']['precision'],
        callbacks=callbacks,
        logger=comet_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # 7. Train
    try:
        trainer.fit(model)
    except Exception as e:
        import traceback
        logger.error(f"Trial {trial.number} failed with exception: {e}")
        logger.error(traceback.format_exc())
        # Clean up temp config file
        trial_config_path.unlink()
        # Report failure to Optuna
        raise optuna.exceptions.TrialPruned()


    # 8. Return metric
    # Clean up temp config file
    trial_config_path.unlink()
    
    return checkpoint_callback.best_model_score.item()


def run_optimization(args, base_config_path: Path) -> Dict[str, float]:
    """
    Run Optuna hyperparameter optimization study.
    """
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install with: pip install optuna optuna-integration")
    # Comet ML setup
    comet_pruner = None
    comet_callback = None
    if COMET_AVAILABLE and os.environ.get('COMET_API_KEY'):
        comet_pruner = CometPruner()
        comet_callback = CometCallback(
            metric_name="val_median_error",
            experiment_name=args.study_name,
        )
        logger.info("Comet ML integration enabled.")
    else:
        logger.warning("Comet ML integration disabled. Set COMET_API_KEY to enable.")
        
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',  # We want to minimize the localization error
        pruner=comet_pruner,
        load_if_exists=True,
    )
    
    # Add objective function as a lambda to pass extra arguments
    obj_fn = lambda trial: objective(trial, args, base_config_path)
    
    # Start optimization
    try:
        study.optimize(
            obj_fn,
            n_trials=args.n_trials,
            callbacks=[comet_callback] if comet_callback else [],
        )
    except KeyboardInterrupt:
        logger.info("Optimization stopped by user.")
        
    # Print results
    logger.info(f"Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    logger.info(f"  Pruned trials: {len(pruned_trials)}")
    logger.info(f"  Complete trials: {len(complete_trials)}")
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value: {trial.value}")
    
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    return dict(trial.params)


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
        self.train_dataset_paths = []
        self.eval_dataset_path = None
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
        if self.args.skip_scenes:
            logger.info("Skipping scene generation (--skip-scenes)")
            return

        self.log_section("STEP 1: Generate Scenes")

        cmd = [sys.executable, "scripts/scene_generation/generate_scenes.py"]

        if self.args.scene_config:
            logger.info(f"Using scene config file: {self.args.scene_config}")
            with open(self.args.scene_config, 'r') as f:
                config = EasyDict(yaml.safe_load(f)).scene_generation
            cities = getattr(config, "cities", None)
            if cities:
                for city_cfg in cities:
                    city = EasyDict(city_cfg)
                    city_cmd = [sys.executable, "scripts/scene_generation/generate_scenes.py"]

                    if city.get('bounding_box'):
                        city_cmd.extend(["--bbox"] + [str(c) for c in city.bounding_box])
                        scene_slug = city.get('slug') or city.get('name', 'custom_bbox')
                        scene_slug = scene_slug.replace(", ", "_").replace(" ", "_").lower()
                    elif city.get('name'):
                        city_cmd.extend(["--area", city.name])
                        scene_slug = city.name.replace(", ", "_").replace(" ", "_").lower()
                    else:
                        raise RuntimeError("Each city entry needs name or bounding_box")

                    scene_dir = self.project_root / "data" / "scenes" / scene_slug
                    city_cmd.extend(["--output", str(scene_dir)])

                    tiles_cfg = city.get('tiles') or config.city.tiles
                    if tiles_cfg and tiles_cfg.num_tiles > 0:
                        city_cmd.append("--tiles")
                        city_cmd.extend(["--tile-size", str(tiles_cfg.tile_size_m)])
                        city_cmd.extend(["--overlap", str(tiles_cfg.overlap_m)])

                    sites_cfg = city.get('sites') or config.sites
                    if sites_cfg:
                        city_cmd.extend(["--num-tx", str(sites_cfg.num_sites_per_tile)])
                        city_cmd.extend(["--site-strategy", sites_cfg.placement_strategy])

                    if self.args.clean and scene_dir.exists():
                        logger.info(f"Cleaning existing scenes at {scene_dir}")
                        shutil.rmtree(scene_dir)

                    self.run_command(city_cmd, f"Scene Generation ({scene_slug})")
                return

            if config.city.name:
                cmd.extend(["--area", config.city.name])
            elif config.city.bounding_box:
                cmd.extend(["--bbox"] + [str(c) for c in config.city.bounding_box])
            
            self.scene_dir = self.project_root / "data" / "scenes" / config.city.name.replace(", ", "_").replace(" ", "_").lower()
            cmd.extend(["--output", str(self.scene_dir)])

            tiles_cfg = config.city.tiles
            if tiles_cfg and tiles_cfg.num_tiles > 0:
                cmd.append("--tiles")
                cmd.extend(["--tile-size", str(tiles_cfg.tile_size_m)])
                cmd.extend(["--overlap", str(tiles_cfg.overlap_m)])

            if config.sites:
                cmd.extend(["--num-tx", str(config.sites.num_sites_per_tile)])
                cmd.extend(["--site-strategy", config.sites.placement_strategy])

        else:
            logger.info(f"Using GIS bounding box: {self.args.bbox} for scene generation")
            cmd.extend([
                "--bbox",
                str(self.args.bbox[0]),  # west
                str(self.args.bbox[1]),  # south
                str(self.args.bbox[2]),  # east
                str(self.args.bbox[3]),  # north
                "--output", str(self.scene_dir),
                "--num-tx", str(self.args.num_tx),
                "--site-strategy", self.args.site_strategy,
            ])
            if self.args.tiles:
                cmd.append("--tiles")
        
        # Clean existing scenes if requested
        if self.args.clean and self.scene_dir.exists():
            logger.info(f"Cleaning existing scenes at {self.scene_dir}")
            shutil.rmtree(self.scene_dir)

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
            if self.args.train_data_configs or self.args.train_datasets or self.args.eval_data_config:
                for config_path in self.args.train_data_configs or []:
                    dataset_path = self._resolve_data_output_path(config_path)
                    if dataset_path is None:
                        raise RuntimeError(f"Could not resolve output path for: {config_path}")
                    if not dataset_path.exists():
                        raise RuntimeError(f"No dataset available to reuse at: {dataset_path}")
                    self.train_dataset_paths.append(dataset_path)

                if self.args.eval_data_config:
                    self.eval_dataset_path = self._resolve_data_output_path(self.args.eval_data_config)
                    if self.eval_dataset_path is None:
                        raise RuntimeError(f"Could not resolve output path for: {self.args.eval_data_config}")
                    if not self.eval_dataset_path.exists():
                        raise RuntimeError(f"No dataset available to reuse at: {self.eval_dataset_path}")

                if self.args.train_datasets:
                    self.train_dataset_paths.extend([self.project_root / p for p in self.args.train_datasets])
                if self.args.eval_dataset:
                    self.eval_dataset_path = self.project_root / self.args.eval_dataset

                if self.train_dataset_paths:
                    self.dataset_path = self.train_dataset_paths[0]
                return
            # Attempt to find the dataset path from the most recent data config if available
            if self.args.data_config:
                self.dataset_path = self._resolve_data_output_path(self.args.data_config)
                print(f"Resolved dataset_path: {self.dataset_path}")
                if self.dataset_path is None or not self.dataset_path.exists():
                     raise RuntimeError(f"No dataset available to reuse at: {self.dataset_path}")
            else: # Fallback to original behavior
                zarr_files = sorted(self.dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not zarr_files:
                    raise RuntimeError(f"No dataset available to reuse in: {self.dataset_dir}")
                self.dataset_path = zarr_files[0]
            logger.info(f"Reusing dataset: {self.dataset_path}")
            return

        self.log_section("STEP 2: Generate Dataset")

        if self.args.train_data_configs or self.args.train_datasets or self.args.eval_data_config:
            for config_path in self.args.train_data_configs or []:
                logger.info(f"Using training data config: {config_path}")
                self._run_dataset_generation_for_config(config_path)
                dataset_path = self._resolve_data_output_path(config_path)
                if dataset_path is None:
                    raise RuntimeError(f"Could not resolve output path for: {config_path}")
                if not dataset_path.exists():
                    raise RuntimeError(f"Dataset not created at: {dataset_path}")
                self.train_dataset_paths.append(dataset_path)

            if self.args.eval_data_config:
                logger.info(f"Using eval data config: {self.args.eval_data_config}")
                self._run_dataset_generation_for_config(self.args.eval_data_config)
                self.eval_dataset_path = self._resolve_data_output_path(self.args.eval_data_config)
                if self.eval_dataset_path is None:
                    raise RuntimeError(f"Could not resolve output path for: {self.args.eval_data_config}")
                if not self.eval_dataset_path.exists():
                    raise RuntimeError(f"Dataset not created at: {self.eval_dataset_path}")

            if self.args.train_datasets:
                self.train_dataset_paths.extend([self.project_root / p for p in self.args.train_datasets])
            if self.args.eval_dataset:
                self.eval_dataset_path = self.project_root / self.args.eval_dataset

            if self.train_dataset_paths:
                self.dataset_path = self.train_dataset_paths[0]
                logger.info(f"Training datasets: {[p.name for p in self.train_dataset_paths]}")
            if self.eval_dataset_path:
                logger.info(f"Eval dataset: {self.eval_dataset_path.name}")
            return
        
        else:
            print("In else, data_config:", self.args.data_config)
            if self.args.data_config:
                self.dataset_path = self._resolve_data_output_path(self.args.data_config)
                print("Resolved:", self.dataset_path)
                if self.dataset_path is None or not self.dataset_path.exists():
                     raise RuntimeError(f"No dataset available to reuse at: {self.dataset_path}")
            else:
                zarr_files = sorted(self.dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not zarr_files:
                    raise RuntimeError(f"No dataset available to reuse in: {self.dataset_dir}")
                self.dataset_path = zarr_files[0]
            logger.info(f"Reusing dataset: {self.dataset_path}")
            return
        
        cmd = [sys.executable, "scripts/generate_dataset.py"]

        if self.args.data_config:
            logger.info(f"Using data generation config file: {self.args.data_config}")
            with open(self.args.data_config, 'r') as f:
                config = EasyDict(yaml.safe_load(f)).data_generation
            
            # The scene_dir used for generation is specified inside the data_config file
            scene_dir_from_config = self.project_root / config.scenes.root_dir
            self.dataset_dir = self.project_root / Path(config.output.path).parent
            
            cmd.extend(["--config", str(self.args.data_config)])
            cmd.extend(["--scene-dir", str(scene_dir_from_config)]) # Override scene-dir based on data_config
            cmd.extend(["--output-dir", str(self.dataset_dir)])

        else: # Original behavior
            logger.info(f"Using scenes from {self.scene_dir} with Sionna ray tracing")
            self.dataset_dir = self.project_root / "data" / "processed" / f"{self.args.scene_name}_dataset"
            cmd.extend([
                "--scene-dir", str(self.scene_dir),
                "--output-dir", str(self.dataset_dir),
                "--carrier-freq", str(self.args.carrier_freq),
                "--bandwidth", str(self.args.bandwidth),
                "--num-ue", str(self.args.num_ues),
            ])

        if self.args.num_scenes:
            cmd.extend(["--num-scenes", str(self.args.num_scenes)])

        # Clean existing dataset if requested
        if self.args.clean and self.dataset_dir.exists():
            logger.info(f"Cleaning existing dataset at {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.run_command(cmd, "Dataset Generation")

        # Verify dataset was created
        if self.args.data_config:
            with open(self.args.data_config, 'r') as f:
                config = yaml.safe_load(f)
            if 'data_generation' in config:
                self.dataset_path = self.project_root / config['data_generation']['output']['path']
            else:
                output_dir = self.project_root / config.get('output_dir')
                self.dataset_path = self._latest_dataset_in_dir(output_dir)
                if self.dataset_path is None:
                    raise RuntimeError(f"No dataset created in: {output_dir}")
        else:
            zarr_files = sorted(self.dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not zarr_files:
                raise RuntimeError(f"No dataset created in: {self.dataset_dir}")
            self.dataset_path = zarr_files[0]

        if not self.dataset_path.exists():
            raise RuntimeError(f"Dataset not created at: {self.dataset_path}")
            
        logger.info(f"Created dataset: {self.dataset_path.name}")
        
        if self.args.eval_dataset:
            self.eval_dataset_path = self.project_root / self.args.eval_dataset
        
    def step_3_train_model(self):
        """Train the transformer model"""
        if self.args.skip_training:
            logger.info("Skipping training (--skip-training)")
            return
        
        step_label = "STEP 4: Train Model" if self.args.optimize else "STEP 3: Train Model"
        self.log_section(step_label)
        
        # Use existing config or create custom one
        if self.optuna_config_path:
            config_path = self.optuna_config_path
        elif self.args.config:
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

    def _apply_optuna_params(self, config: Dict, params: Dict[str, float]) -> Dict:
        if not params:
            return config

        config['model']['radio_encoder']['d_model'] = params['radio_d_model']
        config['model']['radio_encoder']['nhead'] = params['radio_nhead']
        config['model']['radio_encoder']['num_layers'] = params['radio_num_layers']
        config['model']['radio_encoder']['dropout'] = params['radio_dropout']

        config['model']['map_encoder']['d_model'] = params['map_d_model']
        config['model']['map_encoder']['nhead'] = params['map_nhead']
        config['model']['map_encoder']['num_layers'] = params['map_num_layers']
        config['model']['map_encoder']['dropout'] = params['map_dropout']

        config['model']['fusion']['d_fusion'] = params['fusion_d_fusion']
        config['model']['fusion']['nhead'] = params['fusion_nhead']
        config['model']['fusion']['dropout'] = params['fusion_dropout']

        config['training']['learning_rate'] = params['learning_rate']
        config['training']['batch_size'] = params['batch_size']
        return config

    def step_3_optimize_model(self):
        """Run Optuna hyperparameter optimization."""
        if not self.args.optimize:
            return

        self.log_section("STEP 3: Optimize Model (Optuna)")

        base_config_path = self._create_training_config()
        self.optuna_params = run_optimization(self.args, base_config_path)

        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = self._apply_optuna_params(config, self.optuna_params)

        self.optuna_config_path = self.checkpoint_dir / "training_config_optuna.yaml"
        with open(self.optuna_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Optuna best config saved to: {self.optuna_config_path}")

    def step_4_eval_model(self):
        """Evaluate trained model on held-out dataset."""
        if self.args.skip_eval:
            logger.info("Skipping evaluation (--skip-eval)")
            return
        if self.eval_dataset_path is None:
            logger.info("No eval dataset provided; skipping evaluation")
            return
        
        step_label = "STEP 5: Evaluate Model" if self.args.optimize else "STEP 4: Evaluate Model"
        self.log_section(step_label)

        ckpt_path = self.checkpoint_dir / "last.ckpt"
        if not ckpt_path.exists():
            ckpt_candidates = sorted(self.checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not ckpt_candidates:
                raise RuntimeError(f"No checkpoint found in {self.checkpoint_dir} for evaluation")
            ckpt_path = ckpt_candidates[0]

        base_config = self.checkpoint_dir / "training_config.yaml"
        if not base_config.exists():
            raise RuntimeError(f"Training config not found at {base_config}")

        with open(base_config, 'r') as f:
            config = yaml.safe_load(f)
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['test_zarr_paths'] = [str(self.eval_dataset_path)]
        self.eval_config_path = self.checkpoint_dir / "eval_config.yaml"
        with open(self.eval_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", str(self.eval_config_path),
            "--test-only",
            "--resume", str(ckpt_path),
        ]
        self.run_command(cmd, "Model Evaluation")
        
    def _create_training_config(self):
        """Create a custom training config based on pipeline parameters"""
        config_path = self.checkpoint_dir / "training_config.yaml"
        
        # Load base config
        base_config = self.project_root / "configs" / "training" / "training_simple.yaml"
        with open(base_config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Update dataset configuration with all required fields
        if 'dataset' not in config:
            config['dataset'] = {}
        if self.train_dataset_paths:
            config['dataset']['train_zarr_paths'] = [str(p) for p in self.train_dataset_paths]
            config['dataset']['val_zarr_paths'] = [str(p) for p in self.train_dataset_paths]
        else:
            config['dataset']['zarr_path'] = str(self.dataset_path)
        if self.eval_dataset_path:
            config['dataset']['test_zarr_paths'] = [str(self.eval_dataset_path)]
        config['dataset']['map_resolution'] = 2.0  # 2 meters per pixel
        config['dataset']['scene_extent'] = 856  # Scene size in meters
        config['dataset']['normalize_features'] = True
        config['dataset']['handle_missing_values'] = 'mask'
        config['metadata'] = {
            'run_name': self.args.run_name,
            'scene_name': self.args.scene_name,
            'dataset_path': str(self.dataset_path),
            'train_dataset_paths': [str(p) for p in self.train_dataset_paths] if self.train_dataset_paths else None,
            'eval_dataset_path': str(self.eval_dataset_path) if self.eval_dataset_path else None,
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
        if 'checkpoint' not in config['infrastructure']:
            config['infrastructure']['checkpoint'] = {}
        config['infrastructure']['checkpoint']['dirpath'] = str(self.checkpoint_dir)
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
    parser.add_argument('--eval-dataset', type=str, default=None,
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


def main():
    args = parse_args()
    setup_logging(name=args.run_name)

    orchestrator = PipelineOrchestrator(args)
    sys.exit(orchestrator.run())


if __name__ == "__main__":
    main()
