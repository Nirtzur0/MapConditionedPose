"""
Training Script for UE Localization Model

Usage:
    python scripts/train.py --config configs/model.yaml
    python scripts/train.py --config configs/model.yaml --resume checkpoints/last.ckpt
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training import UELocalizationLightning


def main():
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
    print(f"\nModel Parameters: {model.model.num_parameters:,}")
    print(f"Configuration: {args.config}\n")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='model-{epoch:02d}-{val_median_error:.2f}m',
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
    
    # Setup logger
    logger = None
    if config['infrastructure']['logging']['use_wandb']:
        project = args.wandb_project or config['infrastructure']['logging']['project']
        logger = WandbLogger(
            project=project,
            name=args.run_name,
            log_model=False,  # Don't upload checkpoints automatically
        )
        # Log config
        logger.experiment.config.update(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['infrastructure']['accelerator'],
        devices=config['infrastructure']['devices'],
        precision=config['infrastructure']['precision'],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config['training']['gradient_clip'],
        log_every_n_steps=config['infrastructure']['logging']['log_every_n_steps'],
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Test only mode
    if args.test_only:
        if args.resume is None:
            raise ValueError("Must provide --resume checkpoint for test-only mode")
        
        print(f"\nRunning test evaluation on checkpoint: {args.resume}\n")
        trainer.test(model, ckpt_path=args.resume)
        return
    
    # Train
    print("\nStarting training...\n")
    trainer.fit(
        model,
        ckpt_path=args.resume,
    )
    
    # Test on best checkpoint
    print("\nTraining complete. Running final test evaluation...\n")
    trainer.test(
        model,
        ckpt_path='best',
    )
    
    print("\nDone!\n")


if __name__ == '__main__':
    main()
