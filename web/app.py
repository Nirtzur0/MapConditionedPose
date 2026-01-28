"""
UE Localization System Explorer
Streamlit app aligned with current pipeline outputs in outputs/<experiment>/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import yaml
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset

try:
    from src.training import UELocalizationLightning as LitUELocalization
    MODEL_AVAILABLE = True
except Exception as e:
    st.error(f"Model imports failed: {e}")
    MODEL_AVAILABLE = False


st.set_page_config(
    page_title="UE Localization System Explorer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _read_yaml(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _list_experiments(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        return []
    exps = []
    for item in outputs_dir.iterdir():
        if not item.is_dir():
            continue
        if (item / "config.yaml").exists() or (item / "data").exists():
            exps.append(item)
    return sorted(exps, key=lambda p: p.name)


def _find_datasets(exp_dir: Path) -> List[Path]:
    data_dir = exp_dir / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.lmdb"), key=lambda p: p.stat().st_mtime, reverse=True)


def _find_checkpoints(exp_dir: Path) -> List[Path]:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    ckpts = list(ckpt_dir.glob("*.ckpt")) + list(ckpt_dir.glob("*.pth"))
    return sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource
def load_dataset(lmdb_path: str, split: str) -> LMDBRadioLocalizationDataset:
    return LMDBRadioLocalizationDataset(lmdb_path=lmdb_path, split=split)


@st.cache_resource
def load_model(checkpoint_path: str):
    if not MODEL_AVAILABLE:
        return None, None
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_wrapper = LitUELocalization.load_from_checkpoint(ckpt, map_location=device)
        model = model_wrapper.model
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model checkpoint: {e}")
        return None, None


@st.cache_data
def load_positions(lmdb_path: str, split: str) -> Optional[np.ndarray]:
    try:
        dataset = load_dataset(lmdb_path, split)
        if len(dataset) == 0:
            return None
        positions = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            pos = sample["position"].cpu().numpy()
            scene_size = sample.get("scene_size")
            if scene_size is not None:
                if isinstance(scene_size, torch.Tensor):
                    scene_size = scene_size.cpu().numpy()
                scene_size = np.asarray(scene_size)
                if scene_size.shape[0] >= 2:
                    pos = pos * scene_size[:2]
            positions.append(pos)
        return np.asarray(positions)
    except Exception as e:
        st.error(f"Failed to load positions: {e}")
        return None


def _denorm_position(sample: Dict) -> np.ndarray:
    pos = sample["position"].cpu().numpy()
    scene_size = sample.get("scene_size")
    if scene_size is not None:
        if isinstance(scene_size, torch.Tensor):
            scene_size = scene_size.cpu().numpy()
        scene_size = np.asarray(scene_size)
        if scene_size.shape[0] >= 2:
            pos = pos * scene_size[:2]
    return pos


def _plot_positions(positions: np.ndarray, title: str, predictions: Optional[np.ndarray] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode="markers",
        name="Ground Truth",
        marker=dict(size=7, color="#2a9d8f"),
    ))
    if predictions is not None:
        errors = np.linalg.norm(positions - predictions, axis=1)
        for i in range(len(positions)):
            fig.add_trace(go.Scatter(
                x=[positions[i, 0], predictions[i, 0]],
                y=[positions[i, 1], predictions[i, 1]],
                mode="lines",
                line=dict(color="rgba(255, 99, 71, 0.25)", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=predictions[:, 0],
            y=predictions[:, 1],
            mode="markers",
            name="Predictions",
            marker=dict(size=9, color=errors, colorscale="Reds", symbol="x", colorbar=dict(title="Error (m)")),
        ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def _plot_map(map_tensor: torch.Tensor, title: str) -> go.Figure:
    arr = map_tensor.detach().cpu().numpy()
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale="Viridis"))
    fig.update_layout(title=title, height=350, template="plotly_white")
    return fig


def _summarize_measurements(measurements: Dict[str, torch.Tensor]) -> Dict[str, float]:
    stats = {}
    for key, tensor in measurements.items():
        if not torch.is_tensor(tensor):
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            continue
        stats[f"{key}.mean"] = float(np.mean(arr[finite]))
        stats[f"{key}.std"] = float(np.std(arr[finite]))
    return stats


@st.cache_data
def run_predictions(lmdb_path: str, split: str, checkpoint_path: str, num_samples: int, seed: int) -> Optional[Dict]:
    model, device = load_model(checkpoint_path)
    if model is None:
        return None
    dataset = load_dataset(lmdb_path, split)
    if len(dataset) == 0:
        return None

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    gt_positions = []
    predictions = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            gt_pos = _denorm_position(sample)
            gt_positions.append(gt_pos)

            measurements = {k: v.unsqueeze(0).to(device) for k, v in sample["measurements"].items()}
            radio_map = sample.get("radio_map")
            if radio_map is not None:
                radio_map = radio_map.unsqueeze(0).to(device)
            osm_map = sample.get("osm_map")
            if osm_map is not None:
                osm_map = osm_map.unsqueeze(0).to(device)
            scene_idx = sample.get("scene_idx")
            if isinstance(scene_idx, torch.Tensor):
                scene_idx = scene_idx.unsqueeze(0).to(device)

            output = model(measurements, radio_map, osm_map, scene_idx=scene_idx)
            pred_pos = output["predicted_position"].cpu().numpy()[0]
            scene_size = sample.get("scene_size")
            if scene_size is not None:
                if isinstance(scene_size, torch.Tensor):
                    scene_size = scene_size.cpu().numpy()
                scene_size = np.asarray(scene_size)
                if scene_size.shape[0] >= 2:
                    pred_pos = pred_pos * scene_size[:2]
            predictions.append(pred_pos)

    return {
        "indices": indices,
        "gt_positions": np.array(gt_positions),
        "predictions": np.array(predictions),
    }


def main() -> None:
    st.title("📡 UE Localization System Explorer")

    outputs_dir = Path(os.environ.get("MCP_OUTPUTS_DIR", "outputs"))
    experiments = _list_experiments(outputs_dir)

    with st.sidebar:
        st.header("Experiment")
        if not experiments:
            st.error(f"No experiments found under {outputs_dir}")
            return

        exp_dir = st.selectbox("Experiment", experiments, format_func=lambda p: p.name)
        config = _read_yaml(exp_dir / "config.yaml")
        report = _read_yaml(exp_dir / "report.yaml")

        st.markdown("---")
        st.header("Dataset")
        datasets = _find_datasets(exp_dir)
        if not datasets:
            st.error("No LMDB datasets found in this experiment")
            return
        dataset_path = st.selectbox("Dataset", datasets, format_func=lambda p: p.name)

        split_options = ["all", "train", "val", "test"]
        split = st.selectbox("Split", split_options, index=0)

        st.markdown("---")
        st.header("Model")
        checkpoints = _find_checkpoints(exp_dir)
        if checkpoints:
            checkpoint_path = st.selectbox("Checkpoint", checkpoints, format_func=lambda p: p.name)
        else:
            checkpoint_path = None
            st.info("No checkpoints found in this experiment")

        st.markdown("---")
        st.header("Prediction Settings")
        num_samples = st.slider("Prediction samples", 10, 200, 50)
        seed = st.number_input("Sampling seed", min_value=0, max_value=10_000, value=42)

    tabs = st.tabs(["Overview", "Dataset", "Predictions", "Sample Viewer"])

    with tabs[0]:
        st.subheader("Experiment Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Config**")
            if config:
                st.json(config)
            else:
                st.info("No config.yaml found.")
        with col2:
            st.markdown("**Report**")
            if report:
                st.json(report)
            else:
                st.info("No report.yaml found.")

    with tabs[1]:
        st.subheader("Dataset Summary")
        positions = load_positions(str(dataset_path), split)
        if positions is None:
            st.warning("No positions found in dataset")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Samples", positions.shape[0])
            with col2:
                st.metric("X Range", f"{positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} m")
            with col3:
                st.metric("Y Range", f"{positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} m")
            with col4:
                area = (positions[:, 0].max() - positions[:, 0].min()) * (positions[:, 1].max() - positions[:, 1].min())
                st.metric("Area", f"{area / 1e6:.2f} km^2")

            st.plotly_chart(_plot_positions(positions, "UE Ground Truth Positions"), use_container_width=True)

    with tabs[2]:
        st.subheader("Model Predictions")
        if checkpoint_path is None:
            st.info("Select a checkpoint to run predictions.")
        else:
            with st.spinner("Running predictions..."):
                preds = run_predictions(str(dataset_path), split, str(checkpoint_path), num_samples, seed)
            if preds is None:
                st.error("Failed to run predictions.")
            else:
                gt = preds["gt_positions"]
                pr = preds["predictions"]
                errors = np.linalg.norm(gt - pr, axis=1)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Error", f"{np.median(errors):.2f} m")
                with col2:
                    st.metric("Mean Error", f"{np.mean(errors):.2f} m")
                with col3:
                    st.metric("P90 Error", f"{np.percentile(errors, 90):.2f} m")
                with col4:
                    st.metric("Max Error", f"{np.max(errors):.2f} m")

                st.plotly_chart(
                    _plot_positions(gt, "Predictions vs Ground Truth", predictions=pr),
                    use_container_width=True,
                )

    with tabs[3]:
        st.subheader("Sample Viewer")
        dataset = load_dataset(str(dataset_path), split)
        if len(dataset) == 0:
            st.warning("Dataset is empty")
        else:
            idx = st.slider("Sample index", 0, len(dataset) - 1, 0)
            sample = dataset[idx]
            pos = _denorm_position(sample)
            scene_id = sample.get("scene_id", "")
            scene_idx = sample.get("scene_idx", -1)
            st.write(f"Scene ID: {scene_id} | Scene Index: {scene_idx}")
            st.write(f"Position (m): [{pos[0]:.2f}, {pos[1]:.2f}]")

            radio_map = sample.get("radio_map")
            osm_map = sample.get("osm_map")
            if radio_map is not None and osm_map is not None:
                rm = radio_map.detach().cpu()
                om = osm_map.detach().cpu()
                rm_channel = st.selectbox("Radio Map Channel", list(range(rm.shape[0])), index=0)
                om_channel = st.selectbox("OSM Map Channel", list(range(om.shape[0])), index=0)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(_plot_map(rm[rm_channel], f"Radio Map Channel {rm_channel}"), use_container_width=True)
                with col2:
                    st.plotly_chart(_plot_map(om[om_channel], f"OSM Map Channel {om_channel}"), use_container_width=True)
            else:
                st.info("Maps not available in this sample.")

            if "measurements" in sample:
                stats = _summarize_measurements(sample["measurements"])
                if stats:
                    st.markdown("**Measurement Summary**")
                    st.json(stats)


if __name__ == "__main__":
    main()
