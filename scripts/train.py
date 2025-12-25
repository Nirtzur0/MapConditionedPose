"""
Training Script for UE Localization Model

Usage:
    python scripts/train.py --config configs/model.yaml
    python scripts/train.py --config configs/model.yaml --resume checkpoints/last.ckpt
    
View training:
    Comet ML dashboard: https://www.comet.com/
    (Set COMET_API_KEY environment variable)
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger, CometLogger
from pathlib import Path
import yaml
import os
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training import UELocalizationLightning
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    # Setup logging
    setup_logging(name="training")
    
    parser = argparse.ArgumentParser(description='Train UE Localization Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model configuration YAML'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run test evaluation only'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Weights & Biases project name (overrides config)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Run name for logging'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = UELocalizationLightning(args.config)
    
    # Print model info
    logger.info(f"\nModel Parameters: {model.model.num_parameters:,}")
    logger.info(f"Configuration: {args.config}\n")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_dir = config['infrastructure'].get('checkpoint', {}).get('dirpath', 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}-{train_loss_epoch:.2f}',
        monitor=config['infrastructure']['checkpoint']['monitor'],
        mode=config['infrastructure']['checkpoint']['mode'],
        save_top_k=config['infrastructure']['checkpoint']['save_top_k'],
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if 'early_stopping' in config['infrastructure']:
        early_stop_callback = EarlyStopping(
            monitor=config['infrastructure']['early_stopping']['monitor'],
            patience=config['infrastructure']['early_stopping']['patience'],
            mode=config['infrastructure']['early_stopping']['mode'],
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Device stats monitor - commented out to reduce clutter in Comet ML
    # device_monitor = DeviceStatsMonitor()
    # callbacks.append(device_monitor)
    
    # Setup loggers
    loggers = []
    
    # Comet ML (primary logger)
    if config['infrastructure']['logging'].get('use_comet', False):
        comet_api_key = os.environ.get('COMET_API_KEY')
        if comet_api_key:
            project = os.environ.get('COMET_PROJECT_NAME') or config['infrastructure']['logging'].get('project', 'ue-localization')
            comet_logger = CometLogger(
                api_key=comet_api_key,
                project=project,
                workspace=os.environ.get('COMET_WORKSPACE'),
            )
            if args.run_name:
                comet_logger.experiment.set_name(args.run_name)
            else:
                # Auto-generate experiment name with scene/model info
                dataset_path = config.get('dataset', {}).get('zarr_path', '')
                
                # Extract scene name more intelligently from dataset path
                scene_name = "unknown"
                path_parts = Path(dataset_path).parts
                for part in reversed(path_parts):
                    # Look for scene-like names in path components
                    part_lower = part.lower()
                    
                    # Skip technical directory names
                    if (part_lower in ['data', 'processed', 'dataset', 'zarr', 'datasets'] or 
                        part.startswith('dataset_') or 
                        '.zarr' in part_lower or 
                        part_lower.endswith('.zarr')):
                        continue
                        
                    if any(city in part_lower for city in ['austin', 'boulder', 'chicago', 'los_angeles', 'new_york', 'seattle']):
                        # Extract city name
                        for city in ['austin', 'boulder', 'chicago', 'los_angeles', 'new_york', 'seattle']:
                            if city in part_lower:
                                scene_name = city.replace('_', '').replace('los', 'la').replace('new', 'ny')
                                break
                        break
                    elif 'sionna' in part_lower:
                        scene_name = "sionna"
                        break
                    elif len(part) > 3 and not part.startswith(('20', '19')):  # Skip date-like names
                        # Use the first meaningful directory name
                        scene_name = part_lower.replace('_dataset', '').replace('_', '')
                        break
                
                model_name = config.get('model', {}).get('name', 'unknown')
                exp_name = f"{model_name}_{scene_name}_{Path(args.config).stem}"
                comet_logger.experiment.set_name(exp_name)
            loggers.append(comet_logger)
            logger.info(f"\n📊 Comet ML logging enabled")
            logger.info(f"   Project: {project}")
            logger.info(f"   View at: https://www.comet.com/{os.environ.get('COMET_WORKSPACE', 'your-workspace')}/{project}\n")
        else:
            logger.warning("\n⚠️  Comet ML enabled in config but COMET_API_KEY not found")
            logger.warning("   Get your API key from: https://www.comet.com/api/my/settings/")
            logger.warning("   Set it with: export COMET_API_KEY=your-key-here")
            raise ValueError("COMET_API_KEY required when use_comet=true")
    
    # WandB (optional alternative)
    if config['infrastructure']['logging'].get('use_wandb', False):
        project = args.wandb_project or config['infrastructure']['logging']['project']
        wandb_logger = WandbLogger(
            project=project,
            name=args.run_name,
            log_model=False,
        )
        wandb_logger.experiment.config.update(config)
        loggers.append(wandb_logger)
        logger.info(f"📈 WandB logging to project: {project}\n")
    
    # Ensure at least one logger (optional for quick tests)
    if not loggers:
        logger.warning("⚠️  No loggers configured. Running without logging.")
        logger.warning("   Enable use_comet or use_wandb in config for experiment tracking.\n")
        # Use a dummy logger to avoid errors
        from pytorch_lightning.loggers import Logger
        class DummyLogger(Logger):
            @property
            def name(self):
                return "dummy"
            @property  
            def version(self):
                return "0"
            def log_hyperparams(self, params):
                pass
            def log_metrics(self, metrics, step):
                pass
            def finalize(self, status):
                pass
        loggers.append(DummyLogger())

    # Enrich Comet with metadata/assets
    for log in loggers:
        if not hasattr(log, "experiment"):
            continue
        experiment = log.experiment
        if experiment is None:
            continue
        metadata = config.get('metadata', {})
        if metadata:
            experiment.log_parameters(metadata)
        experiment.log_parameter("config_path", str(args.config))
        experiment.log_asset(str(args.config), file_name=Path(args.config).name)
        report_path = Path(args.config).parent / "pipeline_report.json"
        if report_path.exists():
            experiment.log_asset(str(report_path), file_name=report_path.name)
        tags = ["pipeline", "training"]
        if metadata.get("scene_name"):
            tags.append(f"scene:{metadata['scene_name']}")
        if metadata.get("num_tx") is not None:
            tags.append(f"tx:{metadata['num_tx']}")
        if metadata.get("num_trajectories") is not None:
            tags.append(f"traj:{metadata['num_trajectories']}")
        experiment.add_tags(tags)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['infrastructure']['accelerator'],
        devices=config['infrastructure']['devices'],
        precision=config['infrastructure']['precision'],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip'],
        log_every_n_steps=config['infrastructure']['logging']['log_every_n_steps'],
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    # Test only mode
    if args.test_only:
        if args.resume is None:
            raise ValueError("Must provide --resume checkpoint for test-only mode")
        
        logger.info(f"\nRunning test evaluation on checkpoint: {args.resume}\n")
        trainer.test(model, ckpt_path=args.resume)
        return
    
    # Train
    logger.info("\nStarting training...\n")
    trainer.fit(
        model,
        ckpt_path=args.resume,
    )
    
    # Test on best checkpoint
    logger.info("\nTraining complete. Running final test evaluation...\n")
    trainer.test(
        model,
        ckpt_path='best',
    )
    
    logger.info("\nDone!\n")


if __name__ == '__main__':
    main()
