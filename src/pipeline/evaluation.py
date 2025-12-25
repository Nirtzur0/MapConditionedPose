"""
Model Evaluation Pipeline Module

Handles the evaluation of trained models on held-out datasets.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


def evaluate_model(args, checkpoint_dir: Path, eval_dataset_path: Optional[Path],
                  log_section_func, run_command_func):
    """
    Evaluate trained model on held-out dataset.
    """
    if args.skip_eval:
        logger.info("Skipping evaluation (--skip-eval)")
        return
    if eval_dataset_path is None:
        logger.info("No eval dataset provided; skipping evaluation")
        return

    step_label = "STEP 5: Evaluate Model" if args.optimize else "STEP 4: Evaluate Model"
    log_section_func(step_label)

    ckpt_path = checkpoint_dir / "last.ckpt"
    if not ckpt_path.exists():
        ckpt_candidates = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpt_candidates:
            raise RuntimeError(f"No checkpoint found in {checkpoint_dir} for evaluation")
        ckpt_path = ckpt_candidates[0]

    base_config = checkpoint_dir / "training_config.yaml"
    if not base_config.exists():
        raise RuntimeError(f"Training config not found at {base_config}")

    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['test_zarr_paths'] = [str(eval_dataset_path)]
    eval_config_path = checkpoint_dir / "eval_config.yaml"
    with open(eval_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", str(eval_config_path),
        "--test-only",
        "--resume", str(ckpt_path),
    ]
    run_command_func(cmd, "Model Evaluation")
