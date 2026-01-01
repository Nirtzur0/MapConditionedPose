"""
Model Training Pipeline Module

Handles the training of the transformer model.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List

from src.utils.logging_utils import print_info, print_success

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
    if 'weight_decay' in params:
        config['training']['weight_decay'] = params['weight_decay']
    return config


def _resolve_training_datasets(train_dataset_paths: Optional[List[Path]],
                               dataset_path: Optional[Path]) -> List[Path]:
    if train_dataset_paths:
        # Deduplicate while preserving order
        return list(dict.fromkeys(train_dataset_paths))
    if dataset_path:
        return [dataset_path]
    return []


def _dataset_defaults(args) -> Dict:
    return {
        'map_resolution': getattr(args, 'map_resolution', 1.0),
        'scene_extent': getattr(args, 'scene_extent', 512),
        'normalize_features': True,
        'handle_missing_values': 'mask',
    }


def _apply_dataset_paths(dataset_config: Dict, train_paths: List[Path], test_path: Optional[Path]) -> None:
    # 1. Training Paths
    paths = [str(p) for p in train_paths]
    if paths:
        dataset_config['train_zarr_paths'] = paths
        # Map resolution/extent still applies globally for now
    
    # 2. Test Path
    if test_path:
        dataset_config['test_zarr_path'] = str(test_path)
        dataset_config['test_on_eval'] = True
    
    # Clean up old keys
    dataset_config.pop('zarr_path', None)
    dataset_config.pop('zarr_paths', None)
    dataset_config.pop('val_zarr_path', None)
    dataset_config.pop('val_zarr_paths', None)
    dataset_config.pop('train_zarr_path', None)


def _ensure_dataset_config(config: Dict, args, train_paths: List[Path], test_path: Optional[Path]) -> Dict:
    dataset_config = config.setdefault('dataset', {})
    for key, value in _dataset_defaults(args).items():
        dataset_config.setdefault(key, value)
    _apply_dataset_paths(dataset_config, train_paths, test_path)
    return config


def _create_training_config(args, project_root: Path, checkpoint_dir: Path,
                            train_paths: List[Path], test_path: Optional[Path], num_tx: int) -> Path:
    """Create a training config file based on command line args"""
    import yaml

    dataset_config = _dataset_defaults(args)
    _apply_dataset_paths(dataset_config, train_paths, test_path)

    config = {
        'dataset': dataset_config,
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'optimizer': 'adamw',
            'scheduler': 'cosine_with_warmup',
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'gradient_clip_val': 1.0,
            'loss': {
                'coarse_weight': 1.0,
                'fine_weight': 1.0,
                'use_physics_loss': False,
                'augmentation': {
                    'feature_noise': 0.01,
                    'feature_dropout': 0.05,
                    'temporal_dropout': 0.0,
                    'random_flip': True,
                    'random_rotation': True,
                    'scale_range': [0.9, 1.1],
                },
            },
        },
        'infrastructure': {
            'accelerator': 'auto',
            'devices': 1,
            'precision': '16-mixed' if getattr(args, 'fp16', False) else '32-true',
            'num_workers': args.num_workers,
            'checkpoint': {
                'dirpath': str(checkpoint_dir),
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
                'use_comet': args.comet or bool(os.environ.get('COMET_API_KEY')),
                'use_wandb': args.wandb,
                'project': 'ue-localization',
                'log_every_n_steps': 10,
            },
        },
        'model': {
            'name': 'MapConditionedTransformer',
            'radio_encoder': {
                'type': 'SetTransformer',
                'num_cells': num_tx,
                'num_beams': 64,
                'd_model': 128,
                'nhead': 4,
                'num_layers': 4,
                'dropout': 0.1,
                'max_seq_len': 20,
                'rt_features_dim': 16,  # Updated to match dataset loader (radio_dataset.py line 420)
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
                'in_channels': 10,
                'd_model': 256,
                'nhead': 4,
                'num_layers': 4,
                'dropout': 0.1,
                'radio_map_channels': 5,
                'osm_map_channels': 5,
            },
            'fusion': {
                'd_fusion': 128,
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
                'd_hidden': 256,
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
                optuna_params: Optional[Dict], train_dataset_paths: List[Path],
                dataset_path: Optional[Path], eval_dataset_path: Optional[Path], num_tx: int,
                log_section_func, run_command_func):
    """
    Train the transformer model.
    """
    import yaml
    from ..training.optimization import run_optimization

    if args.skip_training:
        print_info("Skipping training")
        return

    step_label = "STEP 4: Train Model" if args.optimize else "STEP 3: Train Model"
    log_section_func(step_label)

    dataset_paths = _resolve_training_datasets(train_dataset_paths, dataset_path)
    eval_path = eval_dataset_path if eval_dataset_path else None

    # Use existing config or create custom one
    if optuna_config_path:
        config_path = optuna_config_path
    elif args.config:
        config_path = args.config
    else:
        if not dataset_paths:
            raise RuntimeError("No training datasets provided. Use --train-datasets or run dataset generation.")
        config_path = _create_training_config(args, project_root, checkpoint_dir, dataset_paths, eval_path, num_tx)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = _ensure_dataset_config(config, args, dataset_paths, eval_path)

    resolved_config_path = checkpoint_dir / "training_config.yaml"
    with open(resolved_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    config_path = resolved_config_path

    if args.optimize and optuna_config_path is None:
        optuna_params = run_optimization(args, config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = _apply_optuna_params(config, optuna_params)
        config = _ensure_dataset_config(config, args, dataset_paths, eval_path)
        optuna_config_path = checkpoint_dir / "training_config.yaml"
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

    print_success(f"Checkpoints: [bold]{checkpoint_dir.relative_to(project_root)}[/bold]")
