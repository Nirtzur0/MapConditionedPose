"""
Data Generation Pipeline Module

Handles the generation of synthetic datasets from scenes using Sionna ray tracing.
"""

import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, List
from easydict import EasyDict
import yaml

logger = logging.getLogger(__name__)


def _resolve_data_output_path(config_path: Path, project_root: Path) -> Optional[Path]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_gen = config.get('data_generation', {})
    if data_gen and 'output' in data_gen and 'path' in data_gen['output']:
        dataset_path = project_root / data_gen['output']['path']
        if dataset_path.is_dir() and not (dataset_path / '.zgroup').exists() and not (dataset_path / '.zarray').exists():
            return _latest_dataset_in_dir(dataset_path) or dataset_path
        return dataset_path
    return None


def _latest_dataset_in_dir(output_dir: Path) -> Optional[Path]:
    zarr_files = sorted(output_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zarr_files[0] if zarr_files else None


def _run_dataset_generation_for_config(config_path: Path, project_root: Path, run_command_func):
    cmd = [sys.executable, "scripts/generate_dataset.py"]
    cmd.extend(["--config", str(config_path)])
    run_command_func(cmd, f"Dataset Generation ({config_path.name})")


def generate_dataset(args, project_root: Path, scene_dir: Path, dataset_dir: Path,
                    train_dataset_paths: List[Path], eval_dataset_path: Optional[Path],
                    log_section_func, run_command_func):
    """
    Generate synthetic dataset from scenes using Sionna ray tracing.
    """
    if args.skip_dataset:
        logger.info("Skipping dataset generation (--skip-dataset)")
        if args.train_data_configs or args.train_datasets or args.eval_data_config:
            for config_path in args.train_data_configs or []:
                dataset_path = _resolve_data_output_path(config_path, project_root)
                if dataset_path is None:
                    raise RuntimeError(f"Could not resolve output path for: {config_path}")
                if not dataset_path.exists():
                    raise RuntimeError(f"No dataset available to reuse at: {dataset_path}")
                train_dataset_paths.append(dataset_path)

            if args.eval_data_config:
                eval_dataset_path = _resolve_data_output_path(args.eval_data_config, project_root)
                if eval_dataset_path is None:
                    raise RuntimeError(f"Could not resolve output path for: {args.eval_data_config}")
                if not eval_dataset_path.exists():
                    raise RuntimeError(f"No dataset available to reuse at: {eval_dataset_path}")

            if args.train_datasets:
                train_dataset_paths.extend([project_root / p for p in args.train_datasets])
            if args.eval_dataset:
                eval_dataset_path = project_root / args.eval_dataset

            return train_dataset_paths[0] if train_dataset_paths else None, eval_dataset_path

        # Attempt to find the dataset path from the most recent data config if available
        if args.data_config:
            dataset_path = _resolve_data_output_path(args.data_config, project_root)
            if dataset_path is None or not dataset_path.exists():
                 raise RuntimeError(f"No dataset available to reuse at: {dataset_path}")
        else: # Fallback to original behavior
            zarr_files = sorted(dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not zarr_files:
                raise RuntimeError(f"No dataset available to reuse in: {dataset_dir}")
            dataset_path = zarr_files[0]
        logger.info(f"Reusing dataset: {dataset_path}")
        return dataset_path, eval_dataset_path

    log_section_func("STEP 2: Generate Dataset")

    if args.train_data_configs or args.train_datasets or args.eval_data_config:
        for config_path in args.train_data_configs or []:
            logger.info(f"Using training data config: {config_path}")
            _run_dataset_generation_for_config(config_path, project_root, run_command_func)
            dataset_path = _resolve_data_output_path(config_path, project_root)
            if dataset_path is None:
                raise RuntimeError(f"Could not resolve output path for: {config_path}")
            if not dataset_path.exists():
                raise RuntimeError(f"Dataset not created at: {dataset_path}")
            train_dataset_paths.append(dataset_path)

        if args.eval_data_config:
            logger.info(f"Using eval data config: {args.eval_data_config}")
            _run_dataset_generation_for_config(args.eval_data_config, project_root, run_command_func)
            eval_dataset_path = _resolve_data_output_path(args.eval_data_config, project_root)
            if eval_dataset_path is None:
                raise RuntimeError(f"Could not resolve output path for: {args.eval_data_config}")
            if not eval_dataset_path.exists():
                raise RuntimeError(f"Dataset not created at: {eval_dataset_path}")

        if args.train_datasets:
            train_dataset_paths.extend([project_root / p for p in args.train_datasets])
        if args.eval_dataset:
            eval_dataset_path = project_root / args.eval_dataset

        dataset_path = train_dataset_paths[0] if train_dataset_paths else None
        logger.info(f"Training datasets: {[p.name for p in train_dataset_paths]}")
        if eval_dataset_path:
            logger.info(f"Eval dataset: {eval_dataset_path.name}")
        return dataset_path, eval_dataset_path

    cmd = [sys.executable, "scripts/generate_dataset.py"]

    if args.data_config:
        logger.info(f"Using data generation config file: {args.data_config}")
        with open(args.data_config, 'r') as f:
            config = EasyDict(yaml.safe_load(f)).data_generation

        # The scene_dir used for generation is specified inside the data_config file
        scene_dir_from_config = project_root / config.scenes.root_dir
        dataset_dir = project_root / Path(config.output.path).parent

        cmd.extend(["--config", str(args.data_config)])
        cmd.extend(["--scene-dir", str(scene_dir_from_config)]) # Override scene-dir based on data_config
        cmd.extend(["--output-dir", str(dataset_dir)])

    else: # Original behavior
        logger.info(f"Using scenes from {scene_dir} with Sionna ray tracing")
        dataset_dir = project_root / "data" / "processed" / f"{args.scene_name}_dataset"
        cmd.extend([
            "--scene-dir", str(scene_dir),
            "--output-dir", str(dataset_dir),
            "--carrier-freq", str(args.carrier_freq),
            "--bandwidth", str(args.bandwidth),
            "--num-ue", str(args.num_ues),
        ])

    if args.num_scenes:
        cmd.extend(["--num-scenes", str(args.num_scenes)])

    # Clean existing dataset if requested
    if args.clean and dataset_dir.exists():
        logger.info(f"Cleaning existing dataset at {dataset_dir}")
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    run_command_func(cmd, "Dataset Generation")

    # Verify dataset was created
    if args.data_config:
        with open(args.data_config, 'r') as f:
            config = yaml.safe_load(f)
        if 'data_generation' in config:
            dataset_path = project_root / config['data_generation']['output']['path']
            # If path points to a directory that contains Zarrs (but isn't one), resolve to latest
            if dataset_path.is_dir() and not (dataset_path / '.zgroup').exists() and not (dataset_path / '.zarray').exists():
                latest = _latest_dataset_in_dir(dataset_path)
                if latest:
                    dataset_path = latest
        else:
            output_dir = project_root / config.get('output_dir')
            dataset_path = _latest_dataset_in_dir(output_dir)
            if dataset_path is None:
                raise RuntimeError(f"No dataset created in: {output_dir}")
    else:
        zarr_files = sorted(dataset_dir.glob("*.zarr"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not zarr_files:
            raise RuntimeError(f"No dataset created in: {dataset_dir}")
        dataset_path = zarr_files[0]

    if not dataset_path.exists():
        raise RuntimeError(f"Dataset not created at: {dataset_path}")

    logger.info(f"Created dataset: {dataset_path.name}")

    if args.eval_dataset:
        eval_dataset_path = project_root / args.eval_dataset

    return dataset_path, eval_dataset_path