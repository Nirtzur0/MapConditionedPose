"""
Optuna Hyperparameter Optimization Module

Handles hyperparameter optimization using Optuna with Comet ML integration.
"""

import os
import logging
from pathlib import Path
from typing import Dict
import yaml
import gc

# Import Comet before torch to avoid warning if possible
try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

import torch

from src.utils.logging_utils import setup_logging
from src.training import UELocalizationLightning

logger = logging.getLogger(__name__)

# Import Optuna and Comet integration if available
import optuna
if COMET_AVAILABLE:
    try:
        from optuna_integration.comet import CometCallback
    except ImportError:
        COMET_AVAILABLE = False
else:
    CometCallback = None

# Debug
logger.info(f"COMET_AVAILABLE: {COMET_AVAILABLE}")
logger.info(f"COMET_API_KEY in env: {'COMET_API_KEY' in os.environ}")

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
    config['model']['radio_encoder']['d_model'] = trial.suggest_categorical('radio_d_model', [64, 128, 256])
    config['model']['radio_encoder']['nhead'] = trial.suggest_categorical('radio_nhead', [2, 4, 8])
    config['model']['radio_encoder']['num_layers'] = trial.suggest_int('radio_num_layers', 2, 6)
    config['model']['radio_encoder']['dropout'] = trial.suggest_float('radio_dropout', 0.1, 0.3)

    # Map Encoder
    config['model']['map_encoder']['d_model'] = trial.suggest_categorical('map_d_model', [128, 256, 384])
    config['model']['map_encoder']['nhead'] = trial.suggest_categorical('map_nhead', [4, 8])
    config['model']['map_encoder']['num_layers'] = trial.suggest_int('map_num_layers', 2, 6)
    config['model']['map_encoder']['dropout'] = trial.suggest_float('map_dropout', 0.1, 0.3)
    config['model']['map_encoder']['patch_size'] = trial.suggest_categorical('map_patch_size', [8, 16])

    # Fusion
    config['model']['fusion']['d_fusion'] = trial.suggest_categorical('fusion_d_fusion', [128, 256, 384])
    config['model']['fusion']['nhead'] = trial.suggest_categorical('fusion_nhead', [4, 8])
    config['model']['fusion']['dropout'] = trial.suggest_float('fusion_dropout', 0.1, 0.3)
    
    # Dependent dimensions - ensure heads match input from fusion
    config['model']['coarse_head']['d_input'] = config['model']['fusion']['d_fusion']
    config['model']['fine_head']['d_input'] = config['model']['fusion']['d_fusion']

    # Training params
    config['training']['learning_rate'] = trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True)
    config['training']['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32])
    config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 0.05, log=True)
    config['training']['warmup_steps'] = trial.suggest_categorical('warmup_steps', [500, 1000, 2000])


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
        project_name = os.environ.get('COMET_PROJECT_NAME') or args.study_name or "ue-localization"
        comet_logger = CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            project_name=project_name,
            workspace=os.environ.get('COMET_WORKSPACE'),
        )
        # Use study name prefix for better organization
        exp_name = f"{args.study_name}_trial_{trial.number}"
        comet_logger.experiment.set_name(exp_name)
        comet_logger.experiment.log_parameters(trial.params)


    trainer = pl.Trainer(
        max_epochs=config['training'].get('num_epochs', 10),
        accelerator=config['infrastructure'].get('accelerator', 'auto'),
        devices=config['infrastructure'].get('devices', 1),
        precision=config['infrastructure'].get('precision', '32-true'),
        callbacks=callbacks,
        logger=comet_logger,
        log_every_n_steps=config['infrastructure'].get('logging', {}).get('log_every_n_steps', 10),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # 7. Train
    try:
        trainer.fit(model)
    except Exception as e:
        import traceback
        logger.error(f"Trial {trial.number} failed with exception: {e}")
        logger.error(traceback.format_exc())
        # Clean up
        if trial_config_path.exists():
            trial_config_path.unlink()
        
        # Memory cleanup
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        # Report failure to Optuna
        raise optuna.exceptions.TrialPruned()

    # 8. Return metric
    # Memory cleanup
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    # Clean up temp config file
    if trial_config_path.exists():
        trial_config_path.unlink()

    return checkpoint_callback.best_model_score.item()

def run_optimization(args, base_config_path: Path) -> Dict[str, float]:
    """
    Run Optuna hyperparameter optimization study.
    """
    if optuna is None:
        raise RuntimeError("Optuna is not installed. Install with: pip install optuna optuna-integration")
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',  # We want to minimize the localization error
        load_if_exists=True,
    )

    # Comet ML setup
    comet_callback = None
    if COMET_AVAILABLE and os.environ.get('COMET_API_KEY'):
        try:
            project_name = os.environ.get('COMET_PROJECT_NAME') or args.study_name or "ue-localization"
            comet_callback = CometCallback(
                study,
                metric_names=["val_median_error"],
                project_name=project_name,
                workspace=os.environ.get('COMET_WORKSPACE')
            )
            logger.info(f"Comet ML integration enabled. Project: {project_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize CometCallback: {e}")
    else:
        logger.warning("Comet ML integration disabled. Set COMET_API_KEY to enable.")

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

    if len(complete_trials) == 0:
        logger.error("No trials completed successfully.")
        return {}
    
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    return dict(trial.params)
