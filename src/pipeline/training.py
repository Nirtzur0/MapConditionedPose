"""
Model Training Pipeline Module

Handles the training of the transformer model.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _apply_optuna_params(config: Dict, params: Dict[str, float]) -> Dict:
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


def _create_training_config(args, project_root: Path, dataset_path: Path, checkpoint_dir: Path, num_tx: int) -> Path:
    """Create a training config file based on command line args"""
    import yaml

    config = {
        'dataset': {
            'path': str(dataset_path),
            'batch_size': args.batch_size,
            'num_workers': 4,
        },
        'training': {
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'weight_decay': 0.01,
            'gradient_clip_val': 1.0,
        },
        'infrastructure': {
            'accelerator': 'gpu' if args.gpu else 'cpu',
            'devices': 1,
            'precision': '16-mixed' if args.fp16 else '32-true',
            'checkpoint': {
                'monitor': 'val_median_error',
                'mode': 'min',
                'save_top_k': 3,
            },
            'early_stopping': {
                'monitor': 'val_median_error',
                'mode': 'min',
                'patience': 10,
            },
            'logging': {
                'use_comet': args.comet,
                'use_wandb': args.wandb,
                'project': 'ue-localization',
            },
        },
        'model': {
            'name': 'MapConditionedTransformer',
            'radio_encoder': {
                'type': 'SetTransformer',
                'num_cells': num_tx,
                'num_beams': 64,
                'd_model': 512,
                'nhead': 8,
                'num_layers': 8,
                'dropout': 0.1,
                'max_seq_len': 20,
                'rt_features_dim': 10,
                'phy_features_dim': 8,
                'mac_features_dim': 6,
                'embedding': {
                    'cell_id_embed_dim': 64,
                    'beam_id_embed_dim': 32,
                    'time_embed_dim': 64,
                    'feature_dim': 256,
                },
                'use_token_masking': True,
                'use_feature_masking': True,
            },
            'map_encoder': {
                'img_size': 256,
                'patch_size': 16,
                'in_channels': 12,
                'd_model': 768,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'radio_map_channels': 7,
                'osm_map_channels': 5,
            },
            'fusion': {
                'd_fusion': 512,
                'nhead': 8,
                'dropout': 0.1,
            },
            'coarse_head': {
                'd_input': 512,
                'grid_size': 32,
                'dropout': 0.1,
            },
            'fine_head': {
                'd_input': 512,
                'top_k': 5,
                'dropout': 0.1,
            },
        },
        'physics_loss': {
            'lambda_phys': 0.1,
            'feature_weights': {
                'path_gain': 1.0,
                'toa': 0.1,
                'aoa': 0.1,
                'snr': 0.5,
                'sinr': 0.5,
                'throughput': 0.2,
                'bler': 0.2,
            },
            'loss_type': 'mse',
            'normalize_features': True,
        },
    }

    config_path = checkpoint_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def train_model(args, project_root: Path, checkpoint_dir: Path, optuna_config_path: Optional[Path],
                optuna_params: Optional[Dict], dataset_path: Path, num_tx: int,
                log_section_func, run_command_func):
    """
    Train the transformer model.
    """
    import yaml
    from .optuna_optimizer import run_optimization

    if args.skip_training:
        logger.info("Skipping training (--skip-training)")
        return

    step_label = "STEP 4: Train Model" if args.optimize else "STEP 3: Train Model"
    log_section_func(step_label)

    # Use existing config or create custom one
    if optuna_config_path:
        config_path = optuna_config_path
    elif args.config:
        config_path = args.config
    else:
        config_path = _create_training_config(args, project_root, dataset_path, checkpoint_dir, num_tx)
        if args.optuna:
            optuna_params = run_optimization(args, config_path)
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config = _apply_optuna_params(config, optuna_params)
            optuna_config_path = checkpoint_dir / "training_config_optuna.yaml"
            with open(optuna_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            config_path = optuna_config_path

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", str(config_path),
    ]

    if args.resume_checkpoint:
        cmd.extend(["--resume", str(args.resume_checkpoint)])

    if args.wandb:
        cmd.extend(["--wandb-project", "transformer-ue-localization"])
        cmd.extend(["--run-name", args.run_name])
    if args.comet:
        cmd.extend(["--run-name", args.run_name])

    run_command_func(cmd, "Model Training")

    logger.info(f"Training complete. Checkpoints saved to {checkpoint_dir}")