"""Plot helpers for notebooks/visualization_presentation.ipynb."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml

from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    gaussian_filter,
)

from src.physics_loss import (
    differentiable_lookup,
    PhysicsLoss,
    PhysicsLossConfig,
    refine_position,
    RefineConfig,
)


def resolve_checkpoint_and_config(
    checkpoint_dir="outputs",
    checkpoint_path="",
    checkpoint_epoch=None,
):
    """Resolve latest checkpoint and matching config.yaml if available."""
    ckpt_path = str(checkpoint_path) if checkpoint_path else ""
    notes = []

    def _resolve_latest_checkpoint(root: str) -> str:
        root_path = Path(root)
        if not root_path.exists():
            return ""
        ckpts = list(root_path.rglob("*.ckpt"))
        if not ckpts:
            return ""
        ckpts.sort(key=lambda p: p.stat().st_mtime)
        return str(ckpts[-1])

    def _resolve_checkpoint_path(checkpoint_dir: str, fallback_path: str, epoch):
        if epoch is None:
            return fallback_path
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            return fallback_path
        epoch_int = int(epoch)
        epoch_str = str(epoch_int)
        candidates = [
            p for p in ckpt_dir.rglob("model-epoch=*.ckpt")
            if f"model-epoch={epoch_str}" in p.name
        ]
        if not candidates:
            candidates = list(ckpt_dir.rglob(f"model-epoch={epoch_int:02d}-*.ckpt"))
        if not candidates:
            return fallback_path
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return str(candidates[-1])

    def _resolve_config_from_checkpoint(ckpt_path: str) -> str:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            return ""
        for parent in ckpt.parents:
            if parent.name == "checkpoints":
                exp_dir = parent.parent
                cfg = exp_dir / "config.yaml"
                if cfg.exists():
                    return str(cfg)
                break
        return ""

    ckpt_path = _resolve_checkpoint_path(checkpoint_dir, ckpt_path, checkpoint_epoch)
    if not ckpt_path:
        ckpt_path = _resolve_latest_checkpoint(checkpoint_dir)
    if not ckpt_path:
        ckpt_path = _resolve_latest_checkpoint("outputs")
    if not ckpt_path:
        ckpt_path = _resolve_latest_checkpoint("checkpoints")

    if ckpt_path:
        cfg = _resolve_config_from_checkpoint(ckpt_path)
    else:
        cfg = ""

    if not ckpt_path:
        notes.append("No checkpoint found; set CHECKPOINT_DIR/CHECKPOINT_PATH.")
    if ckpt_path and not cfg:
        notes.append("Config not found next to checkpoint; falling back to configs/training/training.yaml.")
        cfg = "configs/training/training.yaml"

    return ckpt_path, cfg, notes


def resolve_lmdb_paths(candidate_dirs=None):
    """Resolve latest LMDB dataset and train/val/test paths."""
    if candidate_dirs is None:
        candidate_dirs = [
            Path("outputs/experiment/data"),
            Path("outputs/quick_test/data"),
            Path("data/processed"),
        ]
    data_dir = next((p for p in candidate_dirs if p.exists()), Path("data/processed"))
    possible = sorted(data_dir.glob("*.lmdb"))
    train = sorted(data_dir.glob("*_train.lmdb"))
    val = sorted(data_dir.glob("*_val.lmdb"))
    test = sorted(data_dir.glob("*_test.lmdb"))

    dataset_path = str(possible[-1]) if possible else "data/processed/dataset.lmdb"
    train_path = str(train[-1]) if train else dataset_path
    val_path = str(val[-1]) if val else dataset_path
    test_path = str(test[-1]) if test else dataset_path
    return dataset_path, train_path, val_path, test_path


def load_model_and_batch(
    checkpoint_path: str,
    base_config_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    target_batch_idx: int = 0,
    verbose: bool = False,
):
    """Load Lightning checkpoint and return model, batch, and loader."""
    from src.training import UELocalizationLightning
    import os
    temp_config_path = "notebooks/temp_viz_config.yaml"

    if not checkpoint_path or Path(checkpoint_path).is_dir():
        raise RuntimeError("No checkpoint file found. Set CHECKPOINT_DIR or CHECKPOINT_PATH to a .ckpt file.")
    if not Path(base_config_path).exists():
        raise RuntimeError(f"Config not found: {base_config_path}")

    model = None
    try:
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        dataset_cfg = config.get("dataset", {})
        dataset_cfg["train_lmdb_paths"] = [train_path]
        dataset_cfg["val_lmdb_paths"] = [val_path]
        dataset_cfg["test_lmdb_paths"] = [test_path]
        dataset_cfg["use_lmdb"] = True
        for key in [
            "train_zarr_paths",
            "val_zarr_paths",
            "test_zarr_paths",
            "test_zarr_path",
            "val_zarr_path",
        ]:
            dataset_cfg.pop(key, None)
        config["dataset"] = dataset_cfg

        with open(temp_config_path, "w") as f:
            yaml.safe_dump(config, f)

        model = UELocalizationLightning.load_from_checkpoint(
            checkpoint_path,
            config_path=temp_config_path,
            strict=False,
        )
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    if model is None:
        raise RuntimeError("Model failed to load")

    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    if verbose:
        print(f"Model device: {model.device}")

    val_loader = model.val_dataloader()
    if len(val_loader) == 0:
        raise RuntimeError("val_loader is empty; check dataset paths")

    batch_iter = iter(val_loader)
    batch = None
    for _ in range(target_batch_idx + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
    if batch is None:
        raise RuntimeError("No batches found in val_loader")

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    v[sub_k] = sub_v.to(model.device)

    return model, batch, val_loader


def _save_figure(fig, save_path):
    if not save_path:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')

def _resolve_scene_size(batch, sample_idx=0, default_extent=512.0):
    if 'scene_size' in batch:
        scene_size = batch['scene_size'][sample_idx]
        if torch.is_tensor(scene_size):
            scene_size = scene_size.detach().cpu().numpy()
        return np.array(scene_size, dtype=np.float32)
    for key in ('sample_extent', 'scene_extent'):
        if key in batch:
            extent = batch[key]
            if torch.is_tensor(extent):
                extent_val = extent[sample_idx].item() if extent.dim() > 0 else extent.item()
            else:
                extent_val = float(extent)
            return np.array([extent_val, extent_val], dtype=np.float32)
    return np.array([default_extent, default_extent], dtype=np.float32)

def _resolve_scene_extent_scalar(batch=None, scene_size=None, default_extent=512.0):
    if scene_size is None and batch is not None:
        scene_size = _resolve_scene_size(batch, default_extent=default_extent)
    if scene_size is None:
        return float(default_extent)
    return float(np.max(scene_size))

def _extract_scene_bbox(sample, scene_metadata):
    bbox = scene_metadata.get("bbox") or {}
    if not bbox and "scene_bbox" in sample:
        x_min, y_min, x_max, y_max = sample["scene_bbox"]
        bbox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
    if not bbox:
        return None
    try:
        return {
            "x_min": float(bbox["x_min"]),
            "y_min": float(bbox["y_min"]),
            "x_max": float(bbox["x_max"]),
            "y_max": float(bbox["y_max"]),
        }
    except (KeyError, TypeError, ValueError):
        return None

def _points_look_global(points, bbox, tol=1e-3):
    if points.size == 0 or bbox is None:
        return False
    x_min, x_max = bbox["x_min"], bbox["x_max"]
    y_min, y_max = bbox["y_min"], bbox["y_max"]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (
        mins[0] >= x_min - tol
        and maxs[0] <= x_max + tol
        and mins[1] >= y_min - tol
        and maxs[1] <= y_max + tol
    )

def _points_look_local(points, bbox, tol=1e-3):
    if points.size == 0 or bbox is None:
        return False
    extent_x = bbox["x_max"] - bbox["x_min"]
    extent_y = bbox["y_max"] - bbox["y_min"]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (
        mins[0] >= -tol
        and mins[1] >= -tol
        and maxs[0] <= extent_x + tol
        and maxs[1] <= extent_y + tol
    )

def _points_look_normalized(points, bbox):
    if points.size == 0 or bbox is None:
        return False
    extent_x = bbox["x_max"] - bbox["x_min"]
    extent_y = bbox["y_max"] - bbox["y_min"]
    if max(extent_x, extent_y) < 10.0:
        return False
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins.min() >= -0.1 and maxs.max() <= 1.5

def _coerce_points(points, bbox, coord_mode):
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float32)
    if bbox is None or points.size == 0:
        return points
    extent = np.array([bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"]], dtype=np.float32)
    if coord_mode == "global":
        if _points_look_local(points, bbox):
            points = points + np.array([bbox["x_min"], bbox["y_min"]], dtype=np.float32)
        elif _points_look_normalized(points, bbox):
            points = points * extent + np.array([bbox["x_min"], bbox["y_min"]], dtype=np.float32)
    elif coord_mode == "scene":
        if _points_look_global(points, bbox):
            points = points - np.array([bbox["x_min"], bbox["y_min"]], dtype=np.float32)
        elif _points_look_normalized(points, bbox):
            points = points * extent
    else:
        if _points_look_global(points, bbox):
            points = points - np.array([bbox["x_min"], bbox["y_min"]], dtype=np.float32)
        elif _points_look_normalized(points, bbox):
            points = points * extent
    return points

def _infer_scene_limits(bbox, coord_mode):
    if bbox is None:
        return None
    if coord_mode == "global":
        return (bbox["x_min"], bbox["x_max"], bbox["y_min"], bbox["y_max"])
    extent_x = bbox["x_max"] - bbox["x_min"]
    extent_y = bbox["y_max"] - bbox["y_min"]
    return (0.0, extent_x, 0.0, extent_y)

def plot_ue_tx_scatter(
    dataset,
    max_samples=None,
    use_global_coords=False,
    scene_id_filter=None,
    save_path=None,
    max_scenes=6,
    coord_mode=None,
    ue_size=20,
    ue_alpha=0.5,
    tx_size=90,
    highlight_sample_idx=None,
):
    """Scatterplot UE positions and transmitter locations for a dataset split."""
    if coord_mode is None:
        coord_mode = "global" if use_global_coords else "scene"
    if coord_mode not in ("scene", "global", "auto"):
        raise ValueError("coord_mode must be one of: 'scene', 'global', 'auto'")
    if not hasattr(dataset, "_ensure_initialized") or not hasattr(dataset, "_load_sample_by_index"):
        raise ValueError("plot_ue_tx_scatter expects an LMDB dataset with raw samples.")

    dataset._ensure_initialized()
    indices = dataset._indices
    if indices is None:
        n_samples = dataset._metadata.get("num_samples", 0) if dataset._metadata else 0
        indices = np.arange(n_samples)

    if max_samples is not None:
        indices = indices[:max_samples]

    ue_positions = []
    tx_positions = []
    scenes_seen = set()
    scene_to_ue = {}
    scene_to_tx = {}
    scene_to_bbox = {}
    
    highlight_pos = None
    highlight_scene_id = None

    with dataset._env.begin() as txn:
        for sample_idx in indices:
            sample = dataset._load_sample_by_index(int(sample_idx), txn=txn)
            if sample is None:
                continue

            scene_id = sample.get("scene_id", "")
            if isinstance(scene_id, bytes):
                scene_id = scene_id.decode("utf-8", errors="ignore")
            if scene_id_filter and scene_id != scene_id_filter:
                continue

            scene_metadata = sample.get("scene_metadata") or {}
            bbox = _extract_scene_bbox(sample, scene_metadata)
            if scene_id not in scene_to_bbox and bbox is not None:
                scene_to_bbox[scene_id] = bbox

            pos = sample.get("position")
            if pos is not None:
                ue_x = float(pos[0])
                ue_y = float(pos[1])
                ue_positions.append((ue_x, ue_y))
                scene_to_ue.setdefault(scene_id, []).append((ue_x, ue_y))
                
                if highlight_sample_idx is not None and int(sample_idx) == int(highlight_sample_idx):
                    highlight_pos = np.array([[ue_x, ue_y]], dtype=np.float32)
                    highlight_scene_id = scene_id

            if scene_id not in scenes_seen:
                tx_local = scene_metadata.get("tx_local") or []
                if tx_local:
                    for site in tx_local:
                        site_pos = site.get("position")
                        if not site_pos:
                            continue
                        tx_x = float(site_pos[0])
                        tx_y = float(site_pos[1])
                        tx_positions.append((tx_x, tx_y))
                        scene_to_tx.setdefault(scene_id, []).append((tx_x, tx_y))
                else:
                    sites = scene_metadata.get("sites") or []
                    for site in sites:
                        site_pos = site.get("position")
                        if not site_pos:
                            continue
                        tx_x = float(site_pos[0])
                        tx_y = float(site_pos[1])
                        tx_positions.append((tx_x, tx_y))
                        scene_to_tx.setdefault(scene_id, []).append((tx_x, tx_y))
                scenes_seen.add(scene_id)

    if not ue_positions and not tx_positions:
        print("No UE or TX positions found to plot.")
        return None

    ue_positions = np.array(ue_positions, dtype=np.float32) if ue_positions else None
    tx_positions = np.array(tx_positions, dtype=np.float32) if tx_positions else None

    # If multiple scenes are present and no specific scene filter, render per-scene subplots.
    scene_ids = [sid for sid in scene_to_ue.keys()]
    if scene_id_filter:
        scene_ids = [scene_id_filter] if scene_id_filter in scene_to_ue else scene_ids
    if len(scene_ids) > 1:
        if max_scenes is not None and len(scene_ids) > max_scenes:
            scene_ids = scene_ids[:max_scenes]
            print(f"plot_ue_tx_scatter: showing first {max_scenes} scenes.")

        n = len(scene_ids)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), squeeze=False)

        for i, scene_id in enumerate(scene_ids):
            ax = axes[i // cols][i % cols]
            bbox = scene_to_bbox.get(scene_id)
            ue = np.array(scene_to_ue.get(scene_id, []), dtype=np.float32)
            tx = np.array(scene_to_tx.get(scene_id, []), dtype=np.float32)
            ue = _coerce_points(ue, bbox, coord_mode)
            tx = _coerce_points(tx, bbox, coord_mode)
            if ue.size:
                ax.scatter(ue[:, 0], ue[:, 1], s=ue_size, alpha=ue_alpha, color="tab:blue", zorder=2)
            if tx.size:
                ax.scatter(
                    tx[:, 0],
                    tx[:, 1],
                    s=tx_size,
                    marker="^",
                    color="tab:red",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )
            # Highlight specific sample if it belongs to this scene
            if highlight_pos is not None and scene_id == highlight_scene_id:
                hp = _coerce_points(highlight_pos, bbox, coord_mode)
                ax.scatter(hp[:, 0], hp[:, 1], s=ue_size*5, marker="*", color="gold", edgecolors="black", zorder=4, label="Sample")
            
            ax.set_title(scene_id, fontsize=11)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            limits = _infer_scene_limits(bbox, coord_mode)
            if limits is not None:
                ax.set_xlim(limits[0], limits[1])
                ax.set_ylim(limits[2], limits[3])

        # Hide unused axes
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis("off")

        plt.tight_layout()
        _save_figure(fig, save_path)
        plt.show()
        return fig

    # Single-scene plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bbox = None
    if scene_ids:
        bbox = scene_to_bbox.get(scene_ids[0])
    if ue_positions is not None:
        ue_positions = _coerce_points(ue_positions, bbox, coord_mode)
        ax.scatter(
            ue_positions[:, 0],
            ue_positions[:, 1],
            s=ue_size,
            alpha=ue_alpha,
            color="tab:blue",
            label="UE positions",
        )
    if tx_positions is not None:
        tx_positions = _coerce_points(tx_positions, bbox, coord_mode)
        ax.scatter(
            tx_positions[:, 0],
            tx_positions[:, 1],
            s=tx_size,
            marker="^",
            color="tab:red",
            edgecolors="black",
            linewidths=0.5,
            label="TX sites",
        )
    
    if highlight_pos is not None:
         hp = _coerce_points(highlight_pos, bbox, coord_mode)
         ax.scatter(
             hp[:, 0],
             hp[:, 1],
             s=ue_size*5,
             marker="*",
             color="gold",
             edgecolors="black",
             label="Selected Sample",
             zorder=4
         )

    coord_label = "Global" if coord_mode == "global" else "Scene-local"
    title = f"UE and TX Locations ({coord_label} coords)"
    if scene_id_filter:
        title += f" | {scene_id_filter}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    limits = _infer_scene_limits(bbox, coord_mode)
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()
    return fig


def plot_ue_trajectories(
    dataset,
    num_trajectories=10,
    split=None,
    scene_id_filter=None,
    coord_mode="auto",
    seed=0,
    save_path=None,
):
    """Plot N UE trajectories from an LMDB dataset."""
    if coord_mode not in ("scene", "global", "auto"):
        raise ValueError("coord_mode must be one of: 'scene', 'global', 'auto'")
    if not hasattr(dataset, "_ensure_initialized") or not hasattr(dataset, "_load_sample_by_index"):
        raise ValueError("plot_ue_trajectories expects an LMDB dataset with raw samples.")

    dataset._ensure_initialized()
    metadata = dataset._metadata or {}
    seq_slices = metadata.get("sequence_slices")
    if not seq_slices:
        raise RuntimeError("No sequence_slices found; dataset may not include trajectories.")

    if split is None:
        split = getattr(dataset, "split", "all") or "all"
    split = split if split in ("train", "val", "test", "all") else "all"

    seq_ids = None
    if split != "all":
        split_seq = metadata.get("split_sequence_indices") or {}
        seq_ids = split_seq.get(split)
    if not seq_ids:
        seq_ids = list(range(len(seq_slices)))

    if scene_id_filter is not None:
        filtered = []
        with dataset._env.begin() as txn:
            for seq_id in seq_ids:
                start, _ = seq_slices[seq_id]
                sample = dataset._load_sample_by_index(int(start), txn=txn)
                scene_id = sample.get("scene_id", "")
                if isinstance(scene_id, bytes):
                    scene_id = scene_id.decode("utf-8", errors="ignore")
                if scene_id == scene_id_filter:
                    filtered.append(seq_id)
        seq_ids = filtered

    if not seq_ids:
        raise RuntimeError("No sequences available for the requested filters.")

    rng = np.random.default_rng(seed)
    num_pick = min(num_trajectories, len(seq_ids))
    chosen = rng.choice(seq_ids, size=num_pick, replace=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette("husl", num_pick)

    with dataset._env.begin() as txn:
        for color, seq_id in zip(colors, chosen):
            start, length = seq_slices[seq_id]
            positions = []
            t_steps = []
            scene_id = None
            bbox = None

            for idx in range(start, start + length):
                sample = dataset._load_sample_by_index(int(idx), txn=txn)
                if not sample:
                    continue
                pos = sample.get("position")
                if pos is None:
                    continue
                positions.append([float(pos[0]), float(pos[1])])
                t_steps.append(sample.get("t_step", idx - start))
                if scene_id is None:
                    scene_id = sample.get("scene_id", None)
                    if isinstance(scene_id, bytes):
                        scene_id = scene_id.decode("utf-8", errors="ignore")
                if bbox is None:
                    bbox = _extract_scene_bbox(sample, sample.get("scene_metadata", {}) or {})

            if not positions:
                continue
            order = np.argsort(np.asarray(t_steps))
            points = np.asarray(positions, dtype=np.float32)[order]
            points = _coerce_points(points, bbox, coord_mode)

            label = f"seq {seq_id}"
            if scene_id is not None:
                label += f" | {scene_id}"
            ax.plot(points[:, 0], points[:, 1], marker="o", markersize=3, linewidth=1.4, color=color, label=label)

    ax.set_title(f"UE Trajectories (n={num_pick})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    _save_figure(fig, save_path)
    return fig


def plot_ue_tx_scatter_multi(
    datasets,
    max_samples=None,
    scene_id_filter=None,
    save_path=None,
    max_scenes=6,
    coord_mode="auto",
    ue_size=22,
    ue_alpha=0.55,
    tx_size=110,
):
    """Overlay UE positions from multiple datasets/configs per scene."""
    if coord_mode not in ("scene", "global", "auto"):
        raise ValueError("coord_mode must be one of: 'scene', 'global', 'auto'")

    if isinstance(datasets, dict):
        dataset_items = list(datasets.items())
    else:
        dataset_items = [(f"config_{i}", ds) for i, ds in enumerate(datasets)]

    prepared = []
    for label, ds in dataset_items:
        if hasattr(ds, "_ensure_initialized"):
            prepared.append((label, ds))
            continue
        if isinstance(ds, (str, Path)):
            from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset

            prepared.append((label, LMDBRadioLocalizationDataset(str(ds), split="all", normalize=False)))
        else:
            raise ValueError(f"Unsupported dataset entry for label '{label}'")

    scene_to_bbox = {}
    scene_to_tx = {}
    scene_to_ue_by_label = {}

    for label, ds in prepared:
        ds._ensure_initialized()
        indices = ds._indices
        if indices is None:
            n_samples = ds._metadata.get("num_samples", 0) if ds._metadata else 0
            indices = np.arange(n_samples)
        if max_samples is not None:
            indices = indices[:max_samples]

        scene_to_ue = {}
        scenes_seen = set()

        with ds._env.begin() as txn:
            for sample_idx in indices:
                sample = ds._load_sample_by_index(int(sample_idx), txn=txn)
                if sample is None:
                    continue

                scene_id = sample.get("scene_id", "")
                if isinstance(scene_id, bytes):
                    scene_id = scene_id.decode("utf-8", errors="ignore")
                if scene_id_filter and scene_id != scene_id_filter:
                    continue

                scene_metadata = sample.get("scene_metadata") or {}
                bbox = _extract_scene_bbox(sample, scene_metadata)
                if scene_id not in scene_to_bbox and bbox is not None:
                    scene_to_bbox[scene_id] = bbox

                pos = sample.get("position")
                if pos is not None:
                    scene_to_ue.setdefault(scene_id, []).append((float(pos[0]), float(pos[1])))

                if scene_id not in scenes_seen:
                    tx_local = scene_metadata.get("tx_local") or []
                    if tx_local:
                        for site in tx_local:
                            site_pos = site.get("position")
                            if not site_pos:
                                continue
                            scene_to_tx.setdefault(scene_id, []).append(
                                (float(site_pos[0]), float(site_pos[1]))
                            )
                    else:
                        sites = scene_metadata.get("sites") or []
                        for site in sites:
                            site_pos = site.get("position")
                            if not site_pos:
                                continue
                            scene_to_tx.setdefault(scene_id, []).append(
                                (float(site_pos[0]), float(site_pos[1]))
                            )
                    scenes_seen.add(scene_id)

        scene_to_ue_by_label[label] = scene_to_ue

    scene_ids = list(scene_to_ue_by_label.values())
    if scene_ids:
        scene_ids = list({sid for mapping in scene_to_ue_by_label.values() for sid in mapping.keys()})
    else:
        scene_ids = []
    if scene_id_filter:
        scene_ids = [scene_id_filter] if scene_id_filter in scene_ids else scene_ids
    if len(scene_ids) > 1:
        if max_scenes is not None and len(scene_ids) > max_scenes:
            scene_ids = scene_ids[:max_scenes]
            print(f"plot_ue_tx_scatter_multi: showing first {max_scenes} scenes.")

    if not scene_ids:
        print("No scenes found to plot.")
        return None

    n = len(scene_ids)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), squeeze=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    for i, scene_id in enumerate(scene_ids):
        ax = axes[i // cols][i % cols]
        bbox = scene_to_bbox.get(scene_id)

        tx = np.array(scene_to_tx.get(scene_id, []), dtype=np.float32)
        tx = _coerce_points(tx, bbox, coord_mode)
        if tx.size:
            ax.scatter(
                tx[:, 0],
                tx[:, 1],
                s=tx_size,
                marker="^",
                color="tab:red",
                edgecolors="black",
                linewidths=0.6,
                label="TX sites",
                zorder=3,
            )

        for idx, (label, _) in enumerate(prepared):
            ue = np.array(scene_to_ue_by_label.get(label, {}).get(scene_id, []), dtype=np.float32)
            ue = _coerce_points(ue, bbox, coord_mode)
            if not ue.size:
                continue
            color = colors[idx % len(colors)]
            ax.scatter(
                ue[:, 0],
                ue[:, 1],
                s=ue_size,
                alpha=ue_alpha,
                color=color,
                label=f"UE ({label})",
                zorder=2,
            )

        ax.set_title(scene_id, fontsize=11)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        limits = _infer_scene_limits(bbox, coord_mode)
        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()
    return fig


def visualize_all_features(batch, sample_idx=0, model=None, verbose=False):
    """Display all input features in structured tables."""
    from IPython.display import display

    measurements = batch['measurements']

    # Extract data for the sample
    mask = measurements['mask'][sample_idx].cpu().numpy()
    seq_len = int(mask.sum())  # Only take valid steps

    if seq_len == 0:
        print("WARNING: No valid measurements in this sample.")
        return None

    # Feature schemas (matching current encoder inputs)
    rt_feature_names = [
        ('ToA', 's'),
        ('unused (mean_path_gain)', ''),
        ('unused (max_path_gain)', ''),
        ('unused (mean_path_delay)', 's'),
        ('unused (num_paths)', ''),
        ('RMS Delay Spread', 's'),
        ('RMS Angular Spread', 'rad'),
        ('Total Path Power', ''),
        ('N Significant Paths', ''),
        ('unused (delay_range)', 's'),
        ('unused (dominant_path_gain)', ''),
        ('unused (dominant_path_delay)', 's'),
        ('Is NLOS', ''),
        ('Doppler Spread', 'Hz'),
        ('Coherence Time', 's'),
        ('reserved_15', ''),
    ]

    phy_feature_names = [
        ('RSRP', 'dBm'),
        ('RSRQ', 'dB'),
        ('SINR', 'dB'),
        ('CQI', ''),
        ('RI', ''),
        ('PMI', ''),
        ('unused (l1_rsrp)', 'dBm'),
        ('Best Beam ID', ''),
    ]

    mac_feature_names = [
        ('Serving Cell ID', ''),
        ('Neighbor Cell ID 1', ''),
        ('Neighbor Cell ID 2', ''),
        ('Timing Advance', ''),
        ('DL Throughput', 'Mbps'),
        ('BLER', ''),
    ]

    rt_dim = measurements['rt_features'].shape[-1]
    phy_dim = measurements['phy_features'].shape[-1]
    mac_dim = measurements['mac_features'].shape[-1]

    if verbose:
        print(
            f"Feature summary (sample {sample_idx}): "
            f"timesteps={seq_len}, RT={rt_dim}, PHY={phy_dim}, MAC={mac_dim}"
        )

    id_data = {
        'Step': list(range(seq_len)),
        'Timestamp': measurements['timestamps'][sample_idx, :seq_len].cpu().numpy(),
        'Cell ID': measurements['cell_ids'][sample_idx, :seq_len].cpu().numpy(),
        'Beam ID': measurements['beam_ids'][sample_idx, :seq_len].cpu().numpy(),
    }
    df_ids = pd.DataFrame(id_data)
    display(df_ids.style.set_caption("Identifiers (Not Normalized)"))

    rt_values = measurements['rt_features'][sample_idx, :seq_len, :].cpu().numpy()
    rt_data = {}
    for i in range(min(rt_dim, len(rt_feature_names))):
        name, unit = rt_feature_names[i]
        # Skip unused features
        if name.startswith('unused') or name.startswith('reserved'):
            continue
        rt_data[f"{name} ({unit})" if unit else name] = rt_values[:, i]

    df_rt = pd.DataFrame(rt_data)
    styled_rt = df_rt.style.background_gradient(cmap='RdYlGn', axis=0, subset=df_rt.columns[1:]).format(precision=4)
    display(styled_rt.set_caption("RT Layer Features"))

    phy_values = measurements['phy_features'][sample_idx, :seq_len, :].cpu().numpy()
    phy_data = {}
    for i in range(min(phy_dim, len(phy_feature_names))):
        name, unit = phy_feature_names[i]
        # Skip unused features
        if name.startswith('unused'):
            continue
        phy_data[f"{name} ({unit})" if unit else name] = phy_values[:, i]

    df_phy = pd.DataFrame(phy_data)
    styled_phy = df_phy.style.background_gradient(cmap='Blues', axis=0, subset=df_phy.columns[1:]).format(precision=4)
    display(styled_phy.set_caption("PHY Layer Features"))

    mac_values = measurements['mac_features'][sample_idx, :seq_len, :].cpu().numpy()
    mac_data = {}
    for i in range(min(mac_dim, len(mac_feature_names))):
        name, unit = mac_feature_names[i]
        # Skip unused features
        if name.startswith('unused'):
            continue
        mac_data[f"{name} ({unit})" if unit else name] = mac_values[:, i]

    df_mac = pd.DataFrame(mac_data)
    styled_mac = df_mac.style.background_gradient(cmap='Oranges', axis=0, subset=df_mac.columns[1:]).format(precision=4)
    display(styled_mac.set_caption("MAC Layer Features"))

    summary_data = {
        'Layer': ['RT', 'PHY', 'MAC'],
        'Feature Count': [len(rt_data), len(phy_data), len(mac_data)],
        'Mean (abs)': [np.abs(np.array(list(rt_data.values()))).mean() if rt_data else 0,
                       np.abs(np.array(list(phy_data.values()))).mean() if phy_data else 0,
                       np.abs(np.array(list(mac_data.values()))).mean() if mac_data else 0],
        'Std (abs)': [np.array(list(rt_data.values())).std() if rt_data else 0,
                      np.array(list(phy_data.values())).std() if phy_data else 0,
                      np.array(list(mac_data.values())).std() if mac_data else 0],
        'Min': [np.array(list(rt_data.values())).min() if rt_data else 0,
                np.array(list(phy_data.values())).min() if phy_data else 0,
                np.array(list(mac_data.values())).min() if mac_data else 0],
        'Max': [np.array(list(rt_data.values())).max() if rt_data else 0,
                np.array(list(phy_data.values())).max() if phy_data else 0,
                np.array(list(mac_data.values())).max() if mac_data else 0],
    }

    df_summary = pd.DataFrame(summary_data)
    display(df_summary.style.set_caption("Summary Statistics (Normalized)"))

    return {
        'identifiers': df_ids,
        'rt_features': df_rt,
        'phy_features': df_phy,
        'mac_features': df_mac,
        'summary': df_summary,
    }


def _prettify_footprint(osm_map):
    height = osm_map[0] if osm_map.shape[0] > 0 else None
    if osm_map.shape[0] >= 5:
        raw = osm_map[2]
    elif osm_map.shape[0] >= 2:
        raw = osm_map[1]
    else:
        return None

    mask = raw > 0.5
    if height is not None and mask.mean() < 0.01:
        mask = height > 0

    # Smooth jagged triangle edges for nicer visualization
    mask = binary_dilation(mask, iterations=1)
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=1)

    soft = gaussian_filter(mask.astype(float), sigma=1.0)
    if soft.max() > 0:
        soft = soft / soft.max()
    return soft


def visualize_maps(batch, sample_idx=0, save_path=None):
    radio_map = batch['radio_map'][sample_idx].cpu().numpy()
    osm_map = batch['osm_map'][sample_idx].cpu().numpy()
    scene_size = _resolve_scene_size(batch, sample_idx=sample_idx)
    extent = [0.0, float(scene_size[0]), 0.0, float(scene_size[1])]

    num_osm_channels = osm_map.shape[0]
    # TODO: Remove material/terrain/footprint channels from the MapEncoder as well (not just plots).
    osm_display = [
        (0, 'Height', 'Greys'),
        (3, 'Road', 'Greys'),
    ]
    osm_display = [(idx, title, cmap) for idx, title, cmap in osm_display if idx < num_osm_channels]

    num_cols = max(5, len(osm_display))
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))

    # Radio maps
    radio_titles = ['Path Gain', 'ToA', 'SNR', 'SINR', 'Throughput']
    for i in range(5):
        im = axes[0, i].imshow(radio_map[i], cmap='viridis', origin='lower', extent=extent)
        axes[0, i].set_title(f"Radio: {radio_titles[i]}")
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

    # OSM maps
    for slot, (idx, title, cmap) in enumerate(osm_display):
        im = axes[1, slot].imshow(osm_map[idx], cmap=cmap, origin='lower', extent=extent)
        axes[1, slot].set_title(f"OSM: {title}")
        axes[1, slot].axis('off')
        plt.colorbar(im, ax=axes[1, slot], fraction=0.046, pad=0.04)

    # Hide unused subplot axes
    for i in range(len(osm_display), num_cols):
        axes[1, i].axis('off')
        axes[1, i].set_title("(Hidden)")
    for i in range(5, num_cols):
        axes[0, i].axis('off')
        axes[0, i].set_title("(Hidden)")

    plt.suptitle(f"Map Layers (Sample {sample_idx})", fontsize=16)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def visualize_lidar_height(batch, sample_idx=0, save_path=None):
    """Visualize the lidar/terrain height channel from the OSM map."""
    osm_map = batch['osm_map'][sample_idx].cpu().numpy()
    scene_size = _resolve_scene_size(batch, sample_idx=sample_idx)
    extent = [0.0, float(scene_size[0]), 0.0, float(scene_size[1])]
    if osm_map.shape[0] == 0:
        print("No OSM channels available.")
        return
    height = osm_map[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(height, cmap='gray', origin='lower', extent=extent)
    ax.set_title('LiDAR Height (OSM channel 0)', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def list_available_scenes(search_roots=None):
    """List available scenes from data/scenes or outputs/*/scenes."""
    if search_roots is None:
        search_roots = [Path("data/scenes"), Path("outputs")]
    scenes = {}
    for root in search_roots:
        if not root.exists():
            continue
        if root.name == "outputs":
            scene_files = sorted(root.glob("*/scenes/**/scene.xml"))
        else:
            scene_files = sorted(root.rglob("scene.xml"))
        for sf in scene_files:
            scene_name = sf.parent.parent.name
            exp_name = None
            parts = sf.parts
            if "outputs" in parts:
                idx = parts.index("outputs")
                if idx + 1 < len(parts):
                    exp_name = parts[idx + 1]
            key = f"{exp_name}/{scene_name}" if exp_name else scene_name
            scenes.setdefault(key, sf)
    return scenes


def visualize_sionna_3d_scene(
    batch,
    sample_idx=0,
    scene_path=None,
    scene_name=None,
    verbose=False,
    use_sionna=False,
):
    """Render a 3D visualization of the Sionna scene with radio map overlay."""
    if not use_sionna:
        if verbose:
            print("Sionna render disabled (use_sionna=False). Using fallback visualization.")
        return visualize_radio_map_3d_fallback(batch, sample_idx)
    try:
        import sionna as sn
        from sionna.rt import Camera, load_scene

        available_scenes = list_available_scenes()

        if scene_path is None:
            if scene_name and scene_name in available_scenes:
                scene_path = available_scenes[scene_name]
                selected_scene = scene_name
                if verbose:
                    print(f"Using specified scene: {scene_name}")
            elif available_scenes:
                selected_scene = sorted(available_scenes.keys())[0]
                scene_path = available_scenes[selected_scene]
                if verbose:
                    print(f"WARN: No scene specified. Defaulting to first available: {selected_scene}")
                    print("Available scenes:", list(available_scenes.keys()))
                    print("To use a different scene, call: visualize_sionna_3d_scene(batch, scene_name='<name>')")
                    print("TIP: Pre-rendered 3D radio maps for ALL scenes are in:")
                    print("   outputs/*/visualizations/3d/3d_visualizations/")
            else:
                print("No scene files found in data/scenes/ or outputs/*/scenes. Showing fallback visualization.")
                return visualize_radio_map_3d_fallback(batch, sample_idx)

        if verbose:
            print(f"Loading Sionna scene from: {scene_path}")
        scene = load_scene(str(scene_path))

        # Get scene bounds
        try:
            bbox = scene.mi_scene.bbox()
            x_min, y_min, z_min = bbox.min.x, bbox.min.y, bbox.min.z
            x_max, y_max, z_max = bbox.max.x, bbox.max.y, bbox.max.z
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            max_dim = max(x_max - x_min, y_max - y_min)
            ground_z = z_min
            if verbose:
                print(
                    f"Scene bounds: ({x_min:.1f}, {y_min:.1f}) to ({x_max:.1f}, {y_max:.1f}), "
                    f"size: {max_dim:.1f}m"
                )
        except Exception as exc:
            print(f"Could not get scene bounds: {exc}")
            cx, cy, ground_z = 0, 0, 0
            max_dim = 500

        # Get true position from batch and convert to world coordinates
        true_pos = batch['position'][sample_idx].cpu().numpy()
        scene_size = _resolve_scene_size(batch, sample_idx=sample_idx)

        ue_x = cx + (true_pos[0] - 0.5) * scene_size[0]
        ue_y = cy + (true_pos[1] - 0.5) * scene_size[1]
        ue_z = ground_z + 1.5

        if verbose:
            print(f"UE position (world coords): ({ue_x:.1f}, {ue_y:.1f}, {ue_z:.1f})")

        rx = sn.rt.Receiver("UE", position=[float(ue_x), float(ue_y), float(ue_z)])
        scene.add(rx)

        # Camera
        iso_dist = max_dim * 0.8
        cam_z = ground_z + max_dim * 0.6

        cam = Camera(
            position=[float(cx - iso_dist), float(cy - iso_dist), float(cam_z)],
            look_at=[float(cx), float(cy), float(ground_z)],
        )

        # Ensure a transmitter
        if len(scene.transmitters) == 0:
            tx = sn.rt.Transmitter("TX_1", position=[float(cx), float(cy), float(ground_z + 30)])
            scene.add(tx)
            if verbose:
                print(f"Added transmitter at ({cx:.1f}, {cy:.1f}, {ground_z + 30:.1f})")

        scene.tx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = sn.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

        if verbose:
            print("Generating radio map for visualization...")
        solver = sn.rt.RadioMapSolver()

        map_size = min(_resolve_scene_extent_scalar(scene_size=scene_size), max_dim)
        cell_size = max(5.0, map_size / 100)

        radio_map = solver(
            scene,
            center=[float(cx), float(cy), float(ground_z + 1.5)],
            size=[float(map_size), float(map_size)],
            cell_size=[cell_size, cell_size],
            orientation=[0.0, 0.0, 0.0],
            max_depth=3,
            diffraction=False,
        )
        if verbose:
            print("Radio map generated successfully.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Render scene
        ax = axes[0]
        ax.set_title("3D Scene (Isometric View)", fontsize=12)
        try:
            if verbose:
                print("Rendering 3D scene...")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            scene.render_to_file(
                camera=cam,
                filename=tmp_path,
                resolution=(512, 384),
            )

            img = plt.imread(tmp_path)
            ax.imshow(img)
            ax.axis('off')

            os.unlink(tmp_path)
            if verbose:
                print("3D scene render successful.")
        except Exception as exc:
            print(f"Plain render failed: {exc}")
            ax.text(
                0.5,
                0.5,
                "3D Render unavailable (Mitsuba/GPU required)",
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_facecolor('lightgray')
            ax.axis('off')

        # Radio map 2D heatmap
        ax = axes[1]
        ax.set_title("Radio Map - Path Gain (dB)", fontsize=12)
        try:
            if verbose:
                print("Extracting radio map for visualization...")
            path_gain = radio_map.path_gain
            if hasattr(path_gain, 'numpy'):
                path_gain = path_gain.numpy()

            path_gain_db = 10 * np.log10(np.maximum(path_gain, 1e-12))

            if path_gain_db.ndim > 2:
                path_gain_2d = np.max(path_gain_db, axis=0)
            else:
                path_gain_2d = path_gain_db

            im = ax.imshow(
                path_gain_2d,
                cmap='viridis',
                origin='lower',
                extent=[cx - map_size / 2, cx + map_size / 2, cy - map_size / 2, cy + map_size / 2],
                vmin=-120,
                vmax=-50,
            )

            ax.scatter(
                [ue_x],
                [ue_y],
                c='lime',
                s=150,
                marker='*',
                edgecolors='black',
                linewidth=1.5,
                zorder=10,
                label='UE Position',
            )

            ax.scatter(
                [cx],
                [cy],
                c='red',
                s=100,
                marker='^',
                edgecolors='black',
                linewidth=1.5,
                zorder=10,
                label='Transmitter',
            )

            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.legend(loc='upper right')
            plt.colorbar(im, ax=ax, label='Path Gain (dB)')
            if verbose:
                print("Radio map visualization successful.")
        except Exception as exc:
            print(f"Radio map visualization failed: {exc}")
            ax.text(
                0.5,
                0.5,
                f"Radio map extraction failed: {str(exc)[:60]}",
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.axis('off')

        scene_name = Path(scene_path).parent.parent.name if scene_path else "Unknown"
        plt.suptitle(
            f"Sionna 3D Scene: {scene_name}\nUE Position: ({ue_x:.1f}, {ue_y:.1f}) m | Scene Size: {max_dim:.0f}m",
            fontsize=14,
        )
        plt.tight_layout()

        from IPython.display import display

        display(fig)
        plt.close(fig)

        try:
            scene.remove("UE")
        except Exception:
            pass

        return radio_map

    except ImportError as exc:
        print(f"Sionna not available: {exc}")
        print("Falling back to matplotlib 3D visualization...")
        return visualize_radio_map_3d_fallback(batch, sample_idx)
    except Exception as exc:
        print(f"Sionna 3D rendering failed: {exc}")
        import traceback

        traceback.print_exc()
        print("Falling back to matplotlib 3D visualization...")
        return visualize_radio_map_3d_fallback(batch, sample_idx)


def visualize_radio_map_3d_fallback(batch, sample_idx=0):
    """Fallback 3D visualization using matplotlib."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    radio_map = batch['radio_map'][sample_idx].cpu().numpy()
    osm_map = batch['osm_map'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()

    h, w = radio_map.shape[-2:]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    path_gain = radio_map[0]
    path_gain_norm = (path_gain - path_gain.min()) / (path_gain.max() - path_gain.min() + 1e-8)

    heights = osm_map[0] if osm_map.shape[0] > 0 else np.zeros((h, w))
    heights_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-8)

    fig = plt.figure(figsize=(16, 6))

    # 1. 3D Surface plot of radio map
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(
        X,
        Y,
        path_gain_norm * 0.3,
        cmap='viridis',
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )
    ax1.scatter(
        [true_pos[0]],
        [true_pos[1]],
        [0.35],
        c='lime',
        s=100,
        marker='*',
        edgecolors='black',
        zorder=10,
        label='UE Position',
    )
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')
    ax1.set_zlabel('Path Gain (normalized)')
    ax1.set_title('Radio Map 3D Surface', fontsize=12)
    ax1.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Path Gain')

    # 2. Combined height + radio map
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(X, Y, heights_norm * 0.5, color='gray', alpha=0.3, linewidth=0.5)
    ax2.plot_surface(
        X,
        Y,
        np.zeros_like(path_gain_norm),
        facecolors=plt.cm.viridis(path_gain_norm),
        alpha=0.7,
        linewidth=0,
    )

    ax2.scatter(
        [true_pos[0]],
        [true_pos[1]],
        [0.05],
        c='lime',
        s=100,
        marker='*',
        edgecolors='black',
        zorder=10,
        label='UE Position',
    )
    ax2.set_xlabel('X (normalized)')
    ax2.set_ylabel('Y (normalized)')
    ax2.set_zlabel('Height (normalized)')
    ax2.set_title('Buildings + Radio Coverage', fontsize=12)
    ax2.view_init(elev=45, azim=-45)
    ax2.legend()

    plt.suptitle(
        f"3D Visualization (Matplotlib Fallback)\nUE: ({true_pos[0]:.3f}, {true_pos[1]:.3f})",
        fontsize=14,
    )
    plt.tight_layout()

    from IPython.display import display

    display(fig)
    plt.close(fig)

    return None


def render_prediction(model, batch, sample_idx=0, verbose=False, save_path=None):
    plt.style.use('default')
    with torch.no_grad():
        outputs = model.model(batch['measurements'], batch['radio_map'], batch['osm_map'])

    top_k_indices = outputs['top_k_indices'][sample_idx]
    top_k_probs = outputs['top_k_probs'][sample_idx].cpu().numpy()
    fine_offsets = outputs['fine_offsets'][sample_idx].cpu().numpy()
    fine_scores = outputs['fine_scores'][sample_idx].cpu().numpy()
    pred_pos = outputs['predicted_position'][sample_idx].cpu().numpy()
    pred_map = outputs['predicted_position_map'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()

    h, w = batch['radio_map'].shape[-2:]
    grid_size = model.model.grid_size

    cell_cols = (top_k_indices.cpu().numpy() % grid_size).astype(float)
    cell_rows = (top_k_indices.cpu().numpy() // grid_size).astype(float)
    centers = np.stack([(cell_cols + 0.5) / grid_size, (cell_rows + 0.5) / grid_size], axis=-1)
    candidates = centers + fine_offsets

    logits = np.log(np.clip(top_k_probs, 1e-8, None)) + fine_scores
    exps = np.exp(logits - logits.max())
    weights = exps / (exps.sum() + 1e-8)

    # Background: use primary radio-map channel with default colormap
    radio_map = batch['radio_map'][sample_idx].cpu().numpy()
    bg = radio_map[0]
    if hasattr(model, "_normalize_map"):
        bg = model._normalize_map(bg)

    true_px = true_pos[0] * w
    true_py = true_pos[1] * h
    pred_px = pred_pos[0] * w
    pred_py = pred_pos[1] * h
    pred_map_px = pred_map[0] * w
    pred_map_py = pred_map[1] * h

    scene_size = _resolve_scene_size(batch, sample_idx=sample_idx)
    error_m = np.linalg.norm((true_pos - pred_pos) * scene_size)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bg, origin='lower', extent=[0, w, 0, h], interpolation='nearest', cmap='viridis')

    cand_px = candidates[:, 0] * w
    cand_py = candidates[:, 1] * h
    sizes = 40 + 260 * (weights / (weights.max() + 1e-8))
    sc = ax.scatter(
        cand_px,
        cand_py,
        c=weights,
        s=sizes,
        cmap='viridis',
        alpha=0.9,
        edgecolors='none',
        label='Top-K hypotheses',
        zorder=8,
    )

    ax.scatter([true_px], [true_py], c='tab:green', s=150, marker='*', label='Ground Truth', zorder=10)
    ax.scatter([pred_px], [pred_py], c='tab:red', s=100, marker='x', label='Soft mean', linewidth=3, zorder=10)
    ax.scatter([pred_map_px], [pred_map_py], c='tab:orange', s=90, marker='o', label='Best hypothesis', zorder=10)

    if verbose:
        order = np.argsort(-weights)
        print(f"Top-K weights (softmax): {np.round(weights, 4)}")
        for idx in order[:min(5, len(weights))]:
            print(
                f"  k={idx:02d} w={weights[idx]:.4f} "
                f"mu=({candidates[idx,0]:.3f},{candidates[idx,1]:.3f})"
            )

    ax.set_title(
        f"Top-K Re-ranking (Sample {sample_idx})\nError: {error_m:.2f} m | True Pos: ({true_pos[0]:.3f}, {true_pos[1]:.3f}) | Pred Pos: ({pred_pos[0]:.3f}, {pred_pos[1]:.3f})",
        fontsize=12,
    )
    ax.legend(loc='upper right', fontsize=11)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Hypothesis weight')
    ax.axis('off')
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    return outputs


def compute_evaluation_metrics(model, batch, default_scene_extent=512.0):
    """Compute evaluation metrics for the batch."""
    with torch.no_grad():
        outputs = model.model(batch['measurements'], batch['radio_map'], batch['osm_map'])

    pred_pos = outputs['predicted_position'].cpu().numpy()
    true_pos = batch['position'].cpu().numpy()

    if 'scene_size' in batch:
        scene_size = batch['scene_size'].cpu().numpy()
        errors_m = np.linalg.norm((pred_pos - true_pos) * scene_size, axis=1)
        avg_extent = np.max(scene_size, axis=1).mean()
    elif 'sample_extent' in batch:
        scene_extent = batch['sample_extent'].cpu().numpy()
        errors_m = np.linalg.norm((pred_pos - true_pos) * scene_extent[:, None], axis=1)
        avg_extent = scene_extent.mean()
    elif 'scene_extent' in batch:
        scene_extent = batch['scene_extent'].cpu().numpy()
        errors_m = np.linalg.norm((pred_pos - true_pos) * scene_extent[:, None], axis=1)
        avg_extent = scene_extent.mean()
    else:
        scene_extent = default_scene_extent
        errors_m = np.linalg.norm((pred_pos - true_pos) * scene_extent, axis=1)
        avg_extent = scene_extent

    metrics = {
        'Mean Error (m)': np.mean(errors_m),
        'Median Error (m)': np.median(errors_m),
        'RMSE (m)': np.sqrt(np.mean(errors_m ** 2)),
        'Std Error (m)': np.std(errors_m),
        '67th Percentile (m)': np.percentile(errors_m, 67),
        '90th Percentile (m)': np.percentile(errors_m, 90),
        '95th Percentile (m)': np.percentile(errors_m, 95),
        'Max Error (m)': np.max(errors_m),
        'Min Error (m)': np.min(errors_m),
    }

    return metrics, errors_m, pred_pos, true_pos, avg_extent


def plot_error_analysis(errors_m, title="Error Distribution", save_path=None):
    """Plot error analysis visualizations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    sns.histplot(errors_m, bins=20, color='steelblue', alpha=0.7, ax=ax)
    ax.axvline(np.median(errors_m), color='red', linestyle='--', linewidth=2, label=f"Median: {np.median(errors_m):.1f}m")
    ax.axvline(np.mean(errors_m), color='orange', linestyle='-', linewidth=2, label=f"Mean: {np.mean(errors_m):.1f}m")
    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Histogram', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sorted_errors = np.sort(errors_m)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    sns.lineplot(x=sorted_errors, y=cdf * 100, linewidth=2, color='steelblue', ax=ax)
    ax.axhline(67, color='green', linestyle='--', alpha=0.7, label='67th percentile')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.7, label='90th percentile')
    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('CDF (%)', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    ax = axes[2]
    sns.boxplot(y=errors_m, color='lightsteelblue', ax=ax)
    ax.set_ylabel('Localization Error (m)', fontsize=12)
    ax.set_title('Error Box Plot', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    stats_text = f"Median: {np.median(errors_m):.1f}m\nRMSE: {np.sqrt(np.mean(errors_m ** 2)):.1f}m"
    ax.text(1.3, np.median(errors_m), stats_text, fontsize=10, verticalalignment='center')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def plot_error_on_building_map(
    model,
    dataloader,
    max_batches=10,
    cmap="viridis",
    vmin=None,
    vmax=None,
    save_path=None,
):
    """
    Plot per-sample localization error as a scatter overlay on the building map.

    Assumes samples in a batch share the same scene map. If multiple scenes appear
    across batches, a subplot is created per scene_idx.
    """
    import torch

    if model is None:
        raise ValueError("plot_error_on_building_map requires a trained model.")

    scene_data = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            outputs = model.model(batch['measurements'], batch['radio_map'], batch['osm_map'])
            pred_pos = outputs['predicted_position'].cpu().numpy()
            true_pos = batch['position'].cpu().numpy()

            if 'scene_size' in batch:
                scene_size = batch['scene_size'].cpu().numpy()
                errors_m = np.linalg.norm((pred_pos - true_pos) * scene_size, axis=1)
            elif 'scene_extent' in batch:
                scene_extent = batch['scene_extent'].cpu().numpy()
                errors_m = np.linalg.norm((pred_pos - true_pos) * scene_extent[:, None], axis=1)
            else:
                errors_m = np.linalg.norm((pred_pos - true_pos) * 512.0, axis=1)

            scene_idx = batch.get('scene_idx', None)
            if scene_idx is None:
                scene_ids = np.zeros(len(errors_m), dtype=int)
            else:
                scene_ids = scene_idx.cpu().numpy().astype(int)

            for s_id in np.unique(scene_ids):
                mask = scene_ids == s_id
                if not np.any(mask):
                    continue

                if s_id not in scene_data:
                    # Take map and scene size from first occurrence
                    first_idx = int(np.where(mask)[0][0])
                    osm_map = batch['osm_map'][first_idx].cpu().numpy()
                    scene_size = _resolve_scene_size(batch, sample_idx=first_idx)
                    scene_data[s_id] = {
                        'positions': [],
                        'errors': [],
                        'osm_map': osm_map,
                        'scene_size': scene_size,
                    }

                scene_data[s_id]['positions'].append(true_pos[mask])
                scene_data[s_id]['errors'].append(errors_m[mask])

    if not scene_data:
        print("No data collected for error visualization.")
        return

    num_scenes = len(scene_data)
    fig, axes = plt.subplots(1, num_scenes, figsize=(7 * num_scenes, 7), squeeze=False)

    for ax, (s_id, data) in zip(axes[0], scene_data.items()):
        positions = np.concatenate(data['positions'], axis=0)
        errors = np.concatenate(data['errors'], axis=0)
        scene_size = data['scene_size']
        osm_map = data['osm_map']

        extent = [0.0, float(scene_size[0]), 0.0, float(scene_size[1])]
        footprint = _prettify_footprint(osm_map)
        if footprint is None:
            footprint = osm_map[0] if osm_map.shape[0] > 0 else np.zeros((256, 256))

        ax.imshow(footprint, cmap='Greys', origin='lower', extent=extent, alpha=0.6)
        sc = ax.scatter(
            positions[:, 0] * scene_size[0],
            positions[:, 1] * scene_size[1],
            c=errors,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=18,
            alpha=0.85,
            edgecolors='none',
        )
        ax.set_title(f"Scene {s_id} | Error Overlay")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal', 'box')

    plt.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label='Error (m)')
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def compute_loss_breakdown(model, batch):
    """Compute coarse and fine loss components."""
    with torch.no_grad():
        outputs = model.model(batch['measurements'], batch['radio_map'], batch['osm_map'])

    targets = {
        'position': batch['position'],
        'cell_grid': batch['cell_grid'],
    }

    loss_weights = {'coarse_weight': 1.0, 'fine_weight': 1.0}
    losses = model.model.compute_loss(outputs, targets, loss_weights)

    return losses, outputs


def diagnose_coarse_heatmap(outputs, batch, sample_idx=0):
    """Report coarse heatmap alignment and entropy diagnostics."""
    coarse_heatmap = outputs['coarse_heatmap'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    true_cell = int(batch['cell_grid'][sample_idx].item())

    grid_size = int(np.sqrt(coarse_heatmap.shape[0])) if coarse_heatmap.ndim == 1 else coarse_heatmap.shape[0]
    if coarse_heatmap.ndim == 1:
        coarse_heatmap = coarse_heatmap.reshape(grid_size, grid_size)

    probs = coarse_heatmap / (coarse_heatmap.sum() + 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(grid_size * grid_size))

    gt_row = true_cell // grid_size
    gt_col = true_cell % grid_size
    pred_row, pred_col = np.unravel_index(np.argmax(coarse_heatmap), coarse_heatmap.shape)

    # Recompute GT cell index from position to check alignment
    pos_row = int(np.clip(np.floor(true_pos[1] * grid_size), 0, grid_size - 1))
    pos_col = int(np.clip(np.floor(true_pos[0] * grid_size), 0, grid_size - 1))
    recomputed_cell = pos_row * grid_size + pos_col

    print("COARSE HEATMAP DIAGNOSTICS")
    print(f"Entropy: {entropy:.3f} (max {max_entropy:.3f})")
    print(f"GT cell (dataset): {true_cell} -> (row {gt_row}, col {gt_col})")
    print(f"GT cell (from position): {recomputed_cell} -> (row {pos_row}, col {pos_col})")
    print(f"Pred peak: (row {pred_row}, col {pred_col})")


def visualize_coarse_heatmap(outputs, batch, sample_idx=0, save_path=None):
    """Visualize the coarse prediction heatmap and ground truth cell."""
    coarse_heatmap = outputs['coarse_heatmap'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    true_cell = batch['cell_grid'][sample_idx].item()

    grid_size = int(np.sqrt(coarse_heatmap.shape[0])) if coarse_heatmap.ndim == 1 else coarse_heatmap.shape[0]

    if coarse_heatmap.ndim == 1:
        coarse_heatmap = coarse_heatmap.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(coarse_heatmap, cmap='viridis', interpolation='nearest', origin='lower')

    gt_row = true_cell // grid_size
    gt_col = true_cell % grid_size
    rect = plt.Rectangle((gt_col - 0.5, gt_row - 0.5), 1, 1, fill=False, edgecolor='lime', linewidth=3)
    ax.add_patch(rect)

    ax.set_title('Coarse Heatmap with Ground Truth Cell', fontsize=12)
    ax.set_xlabel('X Cell')
    ax.set_ylabel('Y Cell')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.imshow(coarse_heatmap, cmap='viridis', interpolation='nearest', origin='lower')

    gt_x = true_pos[0] * grid_size
    gt_y = true_pos[1] * grid_size
    ax.scatter(gt_x, gt_y, c='lime', s=150, marker='*', edgecolors='black', label='Ground Truth', zorder=10)

    ax.set_title('Coarse Heatmap with True Position', fontsize=12)
    ax.set_xlabel('X Cell')
    ax.set_ylabel('Y Cell')
    ax.legend(loc='upper right')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def visualize_feature_histograms(batch, max_dims=12, use_mask=True):
    """Plot histograms for model input features to sanity-check distributions."""
    measurements = batch.get('measurements', {})
    mask = None
    if use_mask and 'mask' in measurements:
        mask = measurements['mask'].cpu().numpy().reshape(-1)

    def _plot_grid(data, title, max_dims_local):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        dims = min(data.shape[-1], max_dims_local)
        cols = min(4, dims)
        rows = int(np.ceil(dims / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)
        flat = data.reshape(-1, data.shape[-1])
        if mask is not None and mask.shape[0] == flat.shape[0]:
            flat = flat[mask]
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i >= dims:
                ax.axis('off')
                continue
            vals = flat[:, i]
            sns.histplot(vals, bins=40, color='steelblue', alpha=0.85, ax=ax)
            ax.set_title(f"{title} dim {i}", fontsize=10)
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


def visualize_feature_violinplots(
    batch,
    max_dims=12,
    use_mask=True,
    save_dir=None,
    save_prefix="feature_violin",
):
    """Violin plots for model input features to check distributions."""
    measurements = batch.get('measurements', {})
    mask = None
    if use_mask and 'mask' in measurements:
        mask = measurements['mask'].cpu().numpy().reshape(-1)

    def _violin_grid(data, title, max_dims_local, save_name=None):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        dims = min(data.shape[-1], max_dims_local)
        flat = data.reshape(-1, data.shape[-1])
        if mask is not None and mask.shape[0] == flat.shape[0]:
            flat = flat[mask]
        df = pd.DataFrame(flat[:, :dims], columns=[f"dim_{i}" for i in range(dims)])
        df_melt = df.melt(var_name='dim', value_name='value')
        fig, ax = plt.subplots(figsize=(max(8, dims * 0.7), 4))
        sns.violinplot(data=df_melt, x='dim', y='value', inner='quartile', cut=0, linewidth=0.8, ax=ax)
        ax.set_title(f"{title} violin", fontsize=11)
        ax.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        if save_dir:
            safe_name = (save_name or title).replace(" ", "_")
            _save_figure(fig, Path(save_dir) / f"{save_prefix}_{safe_name}.png")
        plt.show()

    for key in ('rt_features', 'phy_features', 'mac_features'):
        if key in measurements:
            _violin_grid(measurements[key].cpu().numpy(), key, max_dims, save_name=key)

    if 'cfr_magnitude' in measurements:
        cfr = measurements['cfr_magnitude'].cpu().numpy()
        if mask is not None:
            cfr = cfr.reshape(-1, cfr.shape[-1])
            if mask.shape[0] == cfr.shape[0]:
                cfr = cfr[mask]
            cfr = cfr.reshape(-1)
        else:
            cfr = cfr.reshape(-1)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.violinplot(y=cfr, inner='quartile', cut=0, linewidth=0.8, color='slateblue', ax=ax)
        ax.set_title('cfr_magnitude (flattened)', fontsize=11)
        ax.grid(True, axis='y', alpha=0.2)
        plt.tight_layout()
        if save_dir:
            _save_figure(fig, Path(save_dir) / f"{save_prefix}_cfr_magnitude.png")
        plt.show()

def visualize_fine_refinement(outputs, batch, sample_idx=0, scene_extent=512.0, model=None, verbose=False):
    """Visualize fine refinement outputs (offsets and re-ranking scores)."""
    fine_offsets = outputs['fine_offsets'][sample_idx].cpu().numpy()
    fine_scores = outputs['fine_scores'][sample_idx].cpu().numpy()
    top_k_indices = outputs['top_k_indices'][sample_idx].cpu().numpy()
    top_k_probs = outputs['top_k_probs'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    pred_pos = outputs['predicted_position'][sample_idx].cpu().numpy()

    if model is not None and hasattr(model, 'model'):
        grid_size = model.model.grid_size
    else:
        heatmap = outputs['coarse_heatmap'][sample_idx]
        grid_size = int(np.sqrt(heatmap.numel())) if heatmap.ndim == 1 else heatmap.shape[-1]

    cell_size = 1.0 / grid_size

    if isinstance(scene_extent, (list, tuple, np.ndarray)):
        scene_size = np.array(scene_extent, dtype=np.float32)
        extent_scalar = float(np.max(scene_size))
    else:
        extent_scalar = float(scene_extent)
        scene_size = np.array([extent_scalar, extent_scalar], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    offset_magnitudes = np.linalg.norm(fine_offsets * scene_size[None, :], axis=1)
    df_offsets = pd.DataFrame({
        'component': [f'K={i}' for i in range(len(offset_magnitudes))],
        'offset_m': offset_magnitudes,
    })
    sns.barplot(data=df_offsets, x='component', y='offset_m', color='steelblue', ax=ax)
    ax.set_ylabel('Offset Magnitude (m)', fontsize=12)
    ax.set_title('Fine Offset Magnitudes per Component', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(cell_size * extent_scalar / 2, color='red', linestyle='--', label=f"Half Cell: {cell_size * extent_scalar / 2:.1f}m")
    ax.legend()

    ax = axes[1]
    logits = np.log(np.clip(top_k_probs, 1e-8, None)) + fine_scores
    exps = np.exp(logits - logits.max())
    weights = exps / (exps.sum() + 1e-8)
    df_probs = pd.DataFrame({
        'component': [f'K={i}' for i in range(len(weights))],
        'coarse_prob': top_k_probs,
        're_rank_weight': weights,
    }).melt(id_vars='component', var_name='type', value_name='value')
    sns.barplot(data=df_probs, x='component', y='value', hue='type', palette=['lightseagreen', 'slateblue'], ax=ax)
    ax.set_ylabel('Probability / Weight', fontsize=12)
    ax.set_title('Coarse Prob vs Re-ranked Weights', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Fine Refinement Analysis (Sample {sample_idx})", fontsize=14)
    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"True position: ({true_pos[0]:.4f}, {true_pos[1]:.4f})")
        print(f"Predicted position: ({pred_pos[0]:.4f}, {pred_pos[1]:.4f})")
        print(f"Error: {np.linalg.norm((true_pos - pred_pos) * scene_size):.2f} m")


def demonstrate_bilinear_resampling(batch, sample_idx=0, verbose=False):
    """Demonstrate differentiable bilinear interpolation from radio maps."""
    radio_map = batch['radio_map'][sample_idx:sample_idx + 1]
    true_pos = batch['position'][sample_idx:sample_idx + 1]

    map_extent = (0.0, 0.0, 1.0, 1.0)

    sampled_features = differentiable_lookup(true_pos, radio_map, map_extent)

    if verbose:
        print("DIFFERENTIABLE BILINEAR RESAMPLING")
        print(f"True Position (normalized): [{true_pos[0, 0].item():.4f}, {true_pos[0, 1].item():.4f}]")
        print(f"Radio Map Shape: {radio_map.shape}")
        print("Sampled Features at True Position:")

    feature_names = ['Path Gain', 'ToA', 'AoA', 'SNR', 'SINR']
    if verbose:
        for name, val in zip(feature_names, sampled_features[0].cpu().numpy()):
            print(f"  {name:12s}: {val:.4f}")

    return sampled_features, radio_map


def visualize_bilinear_sampling(batch, outputs, sample_idx=0):
    """Visualize the bilinear sampling process on the radio map."""
    radio_map = batch['radio_map'][sample_idx].cpu().numpy()
    true_pos = batch['position'][sample_idx].cpu().numpy()
    pred_pos = outputs['predicted_position'][sample_idx].cpu().numpy()

    h, w = radio_map.shape[-2:]

    grid_x = np.linspace(0, 1, 50)
    grid_y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(grid_x, grid_y)
    sample_positions = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32)

    radio_map_tensor = batch['radio_map'][sample_idx:sample_idx + 1].expand(len(sample_positions), -1, -1, -1)

    sampled = differentiable_lookup(
        sample_positions.to(batch['radio_map'].device),
        radio_map_tensor,
        (0.0, 0.0, 1.0, 1.0),
    )

    sampled_map = sampled[:, 0].cpu().numpy().reshape(50, 50)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    im = ax.imshow(radio_map[0], cmap='viridis', origin='lower')
    ax.scatter(true_pos[0] * w, true_pos[1] * h, c='lime', s=100, marker='*', edgecolors='black', label='True Pos', zorder=10)
    ax.scatter(pred_pos[0] * w, pred_pos[1] * h, c='red', s=80, marker='x', linewidth=3, label='Pred Pos', zorder=10)
    ax.set_title('Original Radio Map (Path Gain)', fontsize=12)
    plt.colorbar(im, ax=ax)
    ax.legend()

    ax = axes[1]
    im = ax.imshow(sampled_map, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    ax.scatter(true_pos[0], true_pos[1], c='lime', s=100, marker='*', edgecolors='black', label='True Pos', zorder=10)
    ax.scatter(pred_pos[0], pred_pos[1], c='red', s=80, marker='x', linewidth=3, label='Pred Pos', zorder=10)
    ax.set_title('Bilinear Resampled Map', fontsize=12)
    plt.colorbar(im, ax=ax)
    ax.legend()

    ax = axes[2]
    grad_y, grad_x = np.gradient(sampled_map)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    im = ax.imshow(grad_mag, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])

    # Overlay gradient direction field (normalized) for clarity
    step = max(1, sampled_map.shape[0] // 16)
    grid_x = np.linspace(0, 1, sampled_map.shape[1])[::step]
    grid_y = np.linspace(0, 1, sampled_map.shape[0])[::step]
    gx = grad_x[::step, ::step]
    gy = grad_y[::step, ::step]
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
    gx = gx / mag
    gy = gy / mag
    xx, yy = np.meshgrid(grid_x, grid_y)
    ax.quiver(xx, yy, gx, gy, color='white', alpha=0.8, scale=30, width=0.004)

    ax.set_title('Gradient Magnitude + Direction', fontsize=12)
    plt.colorbar(im, ax=ax)

    plt.suptitle('Differentiable Bilinear Resampling', fontsize=14)
    plt.tight_layout()
    plt.show()


def compute_physics_loss_demo(batch, outputs, sample_idx=0, verbose=False):
    """Demonstrate physics loss computation."""
    num_channels = batch['radio_map'].shape[1]
    channel_names = ('path_gain', 'toa', 'aoa', 'snr', 'sinr')[:num_channels]

    config = PhysicsLossConfig(
        feature_weights={
            'path_gain': 1.0,
            'toa': 0.5,
            'aoa': 0.3,
            'snr': 0.8,
            'sinr': 0.8,
        },
        map_extent=(0.0, 0.0, 1.0, 1.0),
        loss_type='mse',
        normalize_features=False,
        channel_names=channel_names,
    )
    physics_loss_fn = PhysicsLoss(config).to(batch['radio_map'].device)

    pred_pos = outputs['predicted_position']
    true_pos = batch['position']

    observed_features = differentiable_lookup(true_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))

    physics_loss_pred = physics_loss_fn(pred_pos, observed_features, batch['radio_map'])
    physics_loss_gt = physics_loss_fn(true_pos, observed_features, batch['radio_map'])

    if verbose:
        print("PHYSICS LOSS COMPUTATION")
        print(f"Physics Loss at Predicted Positions: {physics_loss_pred.item():.6f}")
        print(f"Physics Loss at Ground Truth:        {physics_loss_gt.item():.6f}")
        print(f"Difference:                          {(physics_loss_pred - physics_loss_gt).item():.6f}")

    return physics_loss_pred, physics_loss_gt


def demonstrate_position_refinement(batch, outputs, sample_idx=0, scene_extent=512.0, model=None, verbose=False):
    """Demonstrate inference-time position refinement using physics loss."""
    initial_pos = outputs['predicted_position'].clone()
    true_pos = batch['position']

    observed_features = differentiable_lookup(true_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))

    mixture_params = None
    coarse_conf = None
    if isinstance(outputs, dict) and 'top_k_probs' in outputs:
        coarse_conf = outputs['top_k_probs'].max(dim=-1).values

    refine_config = RefineConfig(
        num_steps=30,
        learning_rate=0.005,
        min_confidence_threshold=0.6,
        density_weight=0.0,
        clip_to_extent=True,
        map_extent=(0.0, 0.0, 1.0, 1.0),
        physics_config=PhysicsLossConfig(
            map_extent=(0.0, 0.0, 1.0, 1.0),
            loss_type='mse',
        ),
    )

    refined_pos, refine_info = refine_position(
        initial_xy=initial_pos,
        observed_features=observed_features,
        radio_maps=batch['radio_map'],
        config=refine_config,
        confidence=coarse_conf,
        mixture_params=mixture_params,
    )

    with torch.no_grad():
        init_sim = differentiable_lookup(initial_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))
        ref_sim = differentiable_lookup(refined_pos, batch['radio_map'], (0.0, 0.0, 1.0, 1.0))
        init_resid = ((init_sim - observed_features) ** 2).mean(dim=1)
        ref_resid = ((ref_sim - observed_features) ** 2).mean(dim=1)
        worse_mask = ref_resid > init_resid
        if worse_mask.any():
            refined_pos = refined_pos.clone()
            refined_pos[worse_mask] = initial_pos[worse_mask]

    if isinstance(scene_extent, (list, tuple, np.ndarray)):
        scene_size = torch.tensor(scene_extent, device=true_pos.device, dtype=true_pos.dtype)
    else:
        scene_size = torch.tensor([scene_extent, scene_extent], device=true_pos.device, dtype=true_pos.dtype)
    initial_errors = torch.norm((initial_pos - true_pos) * scene_size, dim=1).cpu().numpy()
    refined_errors = torch.norm((refined_pos - true_pos) * scene_size, dim=1).cpu().numpy()

    if verbose:
        reverted = int(worse_mask.sum().item())
        print("POSITION REFINEMENT RESULTS")
        print(f"Initial Physics Loss:  {refine_info.get('loss_initial', 0.0):.6f}")
        print(f"Final Physics Loss:    {refine_info.get('loss_final', 0.0):.6f}")
        print(f"Samples Refined:       {refine_info.get('num_refined', 0)}")
        print(f"Samples Reverted:      {reverted}")
        print(f"Initial Mean Error:  {initial_errors.mean():.2f} m")
        print(f"Refined Mean Error:  {refined_errors.mean():.2f} m")
        print(f"Improvement:         {initial_errors.mean() - refined_errors.mean():.2f} m")

    return initial_pos, refined_pos, initial_errors, refined_errors


def visualize_refinement(batch, initial_pos, refined_pos, sample_idx=0, scene_extent=512.0, save_path=None):
    """Visualize the position refinement trajectory."""
    true_pos = batch['position'][sample_idx].cpu().numpy()
    init_pos = initial_pos[sample_idx].cpu().numpy()
    ref_pos = refined_pos[sample_idx].cpu().numpy()

    radio_map = batch['radio_map'][sample_idx, 0].cpu().numpy()
    h, w = radio_map.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(radio_map, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])

    ax.annotate('', xy=(ref_pos[0], ref_pos[1]), xytext=(init_pos[0], init_pos[1]),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

    ax.scatter(true_pos[0], true_pos[1], c='lime', s=150, marker='*', edgecolors='black', label='Ground Truth', zorder=10)
    ax.scatter(init_pos[0], init_pos[1], c='red', s=80, marker='o', edgecolors='black', label='Initial Pred', zorder=9)
    ax.scatter(ref_pos[0], ref_pos[1], c='blue', s=80, marker='s', edgecolors='black', label='Refined Pred', zorder=9)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(f"Position Refinement (Sample {sample_idx})", fontsize=12)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    plt.colorbar(im, ax=ax, label='Path Gain')
    ax.legend(loc='upper right')

    ax = axes[1]

    if isinstance(scene_extent, (list, tuple, np.ndarray)):
        scene_size = np.array(scene_extent, dtype=np.float32)
    else:
        scene_size = np.array([scene_extent, scene_extent], dtype=np.float32)
    init_error = np.linalg.norm((init_pos - true_pos) * scene_size)
    ref_error = np.linalg.norm((ref_pos - true_pos) * scene_size)

    df_err = pd.DataFrame({
        'stage': ['Initial', 'Refined'],
        'error_m': [init_error, ref_error],
    })
    sns.barplot(data=df_err, x='stage', y='error_m', palette=['coral', 'steelblue'], ax=ax)

    improvement = init_error - ref_error
    improvement_pct = (improvement / init_error) * 100 if init_error > 0 else 0

    ax.text(0.5, max(init_error, ref_error) * 0.95, f"Improvement: {improvement:.1f}m ({improvement_pct:.1f}%)",
            ha='center', fontsize=11, color='darkgreen')

    ax.set_ylabel('Error (m)')
    ax.set_title('Error Before vs After Refinement', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def create_summary_visualization(metrics, losses, initial_errors, refined_errors, scene_extent=512.0):
    """Create a comprehensive summary visualization."""
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Metrics table
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    table_data = [
        ['Metric', 'Value'],
        ['Median Error', f"{metrics['Median Error (m)']:.2f} m"],
        ['RMSE', f"{metrics['RMSE (m)']:.2f} m"],
        ['67th Percentile', f"{metrics['67th Percentile (m)']:.2f} m"],
        ['90th Percentile', f"{metrics['90th Percentile (m)']:.2f} m"],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    ax.set_title('Evaluation Metrics', fontsize=14, pad=20)

    # 2. Loss breakdown pie chart
    ax = fig.add_subplot(gs[0, 1])
    loss_values = [losses['coarse_loss'].item(), losses['fine_loss'].item()]
    loss_labels = ['Coarse (CE)', 'Fine (Smooth-L1)']
    colors = ['#FF6B6B', '#4ECDC4']

    ax.pie(
        loss_values,
        labels=loss_labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),
    )
    ax.set_title('Loss Breakdown', fontsize=14)

    # 3. Before/After refinement comparison
    ax = fig.add_subplot(gs[0, 2])

    df_refine = pd.DataFrame({
        'sample': np.arange(len(initial_errors)),
        'Initial': initial_errors,
        'Refined': refined_errors,
    }).melt(id_vars='sample', var_name='stage', value_name='error_m')
    sns.barplot(data=df_refine, x='sample', y='error_m', hue='stage', ax=ax, palette=['coral', 'steelblue'])

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Error (m)')
    ax.set_title('Per-Sample Refinement Improvement', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. CDF comparison
    ax = fig.add_subplot(gs[1, :2])

    sorted_init = np.sort(initial_errors)
    sorted_ref = np.sort(refined_errors)
    cdf = np.arange(1, len(sorted_init) + 1) / len(sorted_init) * 100

    sns.lineplot(x=sorted_init, y=cdf, linewidth=2, label='Initial', color='coral', ax=ax)
    sns.lineplot(x=sorted_ref, y=cdf, linewidth=2, label='After Refinement', color='steelblue', ax=ax)

    ax.axhline(67, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, 67, '67%', ha='right', va='bottom', fontsize=10)
    ax.text(ax.get_xlim()[1] * 0.98, 90, '90%', ha='right', va='bottom', fontsize=10)

    ax.set_xlabel('Localization Error (m)', fontsize=12)
    ax.set_ylabel('CDF (%)', fontsize=12)
    ax.set_title('Error CDF: Before vs After Physics Refinement', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # 5. Model architecture summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    arch_text = (
        "Model Architecture\n"
        "==================\n\n"
        "1. Radio Encoder\n"
        "   - Transformer (CLS pooling)\n\n"
        "2. Map Encoder\n"
        "   - ViT (E2 optional)\n\n"
        "3. Cross-Attention Fusion\n"
        "   - Multi-head attention\n\n"
        "4. Coarse Head\n"
        "   - Grid cell classifier\n\n"
        "5. Fine Head\n"
        "   - Top-K offset + score re-ranking\n\n"
        "6. Physics Regularization\n"
        "   - Differentiable lookup\n"
    )

    ax.text(
        0.1,
        0.95,
        arch_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.suptitle('UE Localization Pipeline - Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
