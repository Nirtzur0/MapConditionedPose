#!/usr/bin/env python3
"""
Plot N UE trajectories from an LMDB dataset produced by the data generation pipeline.

Example:
    python scripts/plot_ue_trajectories.py \
        --lmdb-path data/processed/dataset_train_0001.lmdb \
        --num-trajectories 12 \
        --split train \
        --output outputs/ue_trajectories.html
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _load_metadata(env: lmdb.Environment) -> Dict:
    with env.begin() as txn:
        raw = txn.get(b"__metadata__")
    if raw is None:
        raise RuntimeError("LMDB metadata not found (missing __metadata__).")
    return pickle.loads(raw)


def _load_sample(env: lmdb.Environment, idx: int) -> Dict:
    key_8 = f"sample_{idx:08d}".encode()
    key_6 = f"sample_{idx:06d}".encode()
    with env.begin() as txn:
        value = txn.get(key_8) or txn.get(key_6)
    if value is None:
        raise KeyError(f"Sample {idx} not found in LMDB.")
    return pickle.loads(value)


def _select_sequence_ids(metadata: Dict, split: str) -> List[int]:
    seq_slices = metadata.get("sequence_slices")
    if not seq_slices:
        return []
    if split == "all":
        return list(range(len(seq_slices)))
    split_seq = metadata.get("split_sequence_indices") or {}
    if split in split_seq and split_seq[split] is not None:
        return list(split_seq[split])
    return []


def _filter_sequences_by_scene(
    env: lmdb.Environment,
    seq_ids: List[int],
    seq_slices: List[Tuple[int, int]],
    scene_id: str,
) -> List[int]:
    filtered = []
    for seq_id in seq_ids:
        start, _ = seq_slices[seq_id]
        sample = _load_sample(env, start)
        if str(sample.get("scene_id", "")) == scene_id:
            filtered.append(seq_id)
    return filtered


def _collect_sequences(
    env: lmdb.Environment,
    seq_ids: List[int],
    seq_slices: List[Tuple[int, int]],
    num_trajectories: int,
    seed: int,
    scene_id: Optional[str],
) -> List[Dict]:
    if not seq_ids:
        return []

    rng = np.random.default_rng(seed)
    if scene_id:
        seq_ids = _filter_sequences_by_scene(env, seq_ids, seq_slices, scene_id)
        if not seq_ids:
            raise RuntimeError(f"No sequences found for scene_id='{scene_id}'.")

    count = min(num_trajectories, len(seq_ids))
    chosen = rng.choice(seq_ids, size=count, replace=False)

    sequences = []
    for seq_id in chosen:
        start, length = seq_slices[seq_id]
        positions = []
        t_steps = []
        scene_name = None
        ue_id = None
        for idx in range(start, start + length):
            sample = _load_sample(env, idx)
            pos = sample.get("position")
            if pos is None:
                continue
            positions.append((float(pos[0]), float(pos[1])))
            t_steps.append(sample.get("t_step", idx - start))
            if scene_name is None:
                scene_name = sample.get("scene_id")
            if ue_id is None:
                ue_id = sample.get("ue_id")

        if not positions:
            continue

        order = np.argsort(np.asarray(t_steps))
        ordered_positions = [positions[i] for i in order]
        sequences.append(
            {
                "sequence_id": int(seq_id),
                "scene_id": scene_name,
                "ue_id": ue_id,
                "positions": ordered_positions,
            }
        )
    return sequences


def _plot_sequences(sequences: List[Dict], output_path: Path) -> None:
    fig = go.Figure()
    for seq in sequences:
        pos = np.asarray(seq["positions"], dtype=np.float32)
        label = f"seq {seq['sequence_id']}"
        if seq.get("ue_id") is not None:
            label += f" | ue {seq['ue_id']}"
        if seq.get("scene_id") is not None:
            label += f" | {seq['scene_id']}"

        fig.add_trace(
            go.Scatter(
                x=pos[:, 0],
                y=pos[:, 1],
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        title=f"UE Trajectories (n={len(sequences)})",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template="plotly_white",
        height=800,
        width=900,
        legend=dict(itemsizing="constant"),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".html":
        fig.write_html(str(output_path))
    else:
        try:
            fig.write_image(str(output_path))
        except Exception as exc:
            fallback = output_path.with_suffix(".html")
            logger.warning(
                "Failed to write image (%s). Writing HTML to %s instead.", exc, fallback
            )
            fig.write_html(str(fallback))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot UE trajectories from LMDB dataset")
    parser.add_argument("--lmdb-path", type=str, required=True, help="Path to LMDB dataset")
    parser.add_argument("--num-trajectories", type=int, default=10, help="Number of trajectories")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument("--scene-id", type=str, default=None, help="Filter to a scene_id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/ue_trajectories.html",
        help="Output path (.html or image)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        metadata = _load_metadata(env)
        seq_slices = metadata.get("sequence_slices")
        if not seq_slices:
            raise RuntimeError(
                "Sequence metadata missing. Regenerate dataset with sequence_length/ue_ids enabled."
            )

        seq_ids = _select_sequence_ids(metadata, args.split)
        if not seq_ids:
            raise RuntimeError(
                f"No sequence indices found for split '{args.split}'."
            )

        sequences = _collect_sequences(
            env,
            seq_ids,
            seq_slices,
            args.num_trajectories,
            args.seed,
            args.scene_id,
        )
        if not sequences:
            raise RuntimeError("No trajectories found to plot.")

        output_path = Path(args.output)
        _plot_sequences(sequences, output_path)
        logger.info("Saved trajectory plot to %s", output_path)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
