"""
Cellular Positioning Model Explorer
Simple map-based tool to visualize ground truth vs predictions
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import torch
import zarr
import sys
from typing import Optional, Tuple, Dict

sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="UE Positioning Explorer",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import model components
try:
    from src.models.ue_localization_model import UELocalizationModel
    from src.training import LitUELocalization
    MODEL_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import model: {e}")
    MODEL_AVAILABLE = False


@st.cache_resource
def load_model(checkpoint_path: Path):
    """Load trained model."""
    if not checkpoint_path.exists():
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load Lightning checkpoint
        model_wrapper = LitUELocalization.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        model = model_wrapper.model
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


@st.cache_data
def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ground truth positions and measurements from zarr."""
    try:
        z = zarr.open(dataset_path, 'r')
        ue_x = np.array(z['positions']['ue_x'])
        ue_y = np.array(z['positions']['ue_y'])
        
        # Load RT measurements if available
        if 'rt' in z:
            rt_data = z['rt']
            # Check if it's nested or direct array
            if hasattr(rt_data, 'keys'):
                # It's a group, find the measurement arrays
                measurements = {}
                for key in rt_data.keys():
                    measurements[key] = np.array(rt_data[key])
            else:
                measurements = np.array(rt_data)
        else:
            measurements = None
            
        return ue_x, ue_y, measurements
        
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None, None, None


@st.cache_data
def generate_predictions(_model, _device, dataset_path: str, num_samples: int = 50):
    """Generate predictions from model on dataset samples."""
    if _model is None:
        return None
    
    try:
        # Load dataset
        from src.datasets.radio_dataset import RadioDataset
        from torch.utils.data import DataLoader
        
        dataset = RadioDataset(dataset_path, sequence_length=16)
        
        # Sample subset
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        predictions = []
        uncertainties = []
        gt_positions = []
        
        with torch.no_grad():
            for idx in indices:
                sample = dataset[idx]
                
                # Get ground truth
                gt_pos = sample['position'].numpy()
                gt_positions.append(gt_pos)
                
                # Prepare inputs
                measurements = sample['measurements'].unsqueeze(0).to(_device)
                radio_map = sample['radio_map'].unsqueeze(0).to(_device) if 'radio_map' in sample else None
                osm_map = sample['osm_map'].unsqueeze(0).to(_device) if 'osm_map' in sample else None
                
                # Run model
                output = _model(measurements, radio_map, osm_map)
                
                pred_pos = output['position'].cpu().numpy()[0]
                predictions.append(pred_pos)
                
                if 'uncertainty' in output:
                    unc = output['uncertainty'].cpu().numpy()[0]
                    uncertainties.append(unc)
                    
        return {
            'gt_positions': np.array(gt_positions),
            'predictions': np.array(predictions),
            'uncertainties': np.array(uncertainties) if uncertainties else None,
            'indices': indices
        }
        
    except Exception as e:
        st.error(f"Failed to generate predictions: {e}")
        return None


def create_position_map(gt_positions: np.ndarray, predictions: Optional[np.ndarray] = None, 
                       uncertainties: Optional[np.ndarray] = None, title: str = "Position Map"):
    """Create interactive map showing GT and predictions."""
    
    fig = go.Figure()
    
    # Ground truth points
    fig.add_trace(go.Scatter(
        x=gt_positions[:, 0],
        y=gt_positions[:, 1],
        mode='markers',
        name='Ground Truth',
        marker=dict(
            size=10,
            color='green',
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        hovertemplate='GT: (%{x:.1f}, %{y:.1f})<extra></extra>'
    ))
    
    # Predictions if available
    if predictions is not None:
        # Error lines connecting GT to prediction
        for i in range(len(gt_positions)):
            fig.add_trace(go.Scatter(
                x=[gt_positions[i, 0], predictions[i, 0]],
                y=[gt_positions[i, 1], predictions[i, 1]],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Prediction markers
        errors = np.linalg.norm(gt_positions - predictions, axis=1)
        
        fig.add_trace(go.Scatter(
            x=predictions[:, 0],
            y=predictions[:, 1],
            mode='markers',
            name='Predictions',
            marker=dict(
                size=12,
                color=errors,
                colorscale='Reds',
                symbol='x',
                line=dict(width=1, color='white'),
                colorbar=dict(title="Error (m)")
            ),
            hovertemplate='Pred: (%{x:.1f}, %{y:.1f})<br>Error: %{marker.color:.1f}m<extra></extra>'
        ))
        
        # Uncertainty ellipses if available
        if uncertainties is not None:
            for i in range(len(predictions)):
                # Create ellipse from uncertainty
                theta = np.linspace(0, 2*np.pi, 50)
                unc = uncertainties[i]
                ellipse_x = predictions[i, 0] + unc[0] * np.cos(theta)
                ellipse_y = predictions[i, 1] + unc[1] * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=ellipse_x,
                    y=ellipse_y,
                    mode='lines',
                    line=dict(color='rgba(255,165,0,0.3)', width=1),
                    fill='toself',
                    fillcolor='rgba(255,165,0,0.1)',
                    showlegend=i == 0,
                    name='Uncertainty' if i == 0 else None,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        template='plotly_white',
        height=700,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig


def create_error_distribution(gt_positions: np.ndarray, predictions: np.ndarray):
    """Create error distribution visualization."""
    
    errors = np.linalg.norm(gt_positions - predictions, axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            marker_color='steelblue'
        ))
        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Error (m)",
            yaxis_title="Count",
            template='plotly_white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CDF
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_errors,
            y=cdf * 100,
            mode='lines',
            line=dict(color='steelblue', width=3)
        ))
        
        # Add percentile lines
        for percentile, color in [(50, 'green'), (90, 'orange'), (95, 'red')]:
            val = np.percentile(errors, percentile)
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                         annotation_text=f"P{percentile}: {val:.1f}m")
        
        fig.update_layout(
            title="Cumulative Distribution",
            xaxis_title="Error (m)",
            yaxis_title="Percentile (%)",
            template='plotly_white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📍 UE Positioning Model Explorer")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        checkpoint_path = Path("checkpoints/best_model.pt")
        if checkpoint_path.exists():
            st.success("✅ Model loaded")
            model, device = load_model(checkpoint_path)
        else:
            st.warning("⚠️ No trained model found")
            model, device = None, None
        
        st.markdown("---")
        
        # Dataset selection
        data_dir = Path("data/processed/quick_test_dataset")
        if data_dir.exists():
            datasets = sorted(list(data_dir.glob("*.zarr")))
            if datasets:
                selected_dataset = st.selectbox(
                    "Dataset",
                    datasets,
                    format_func=lambda x: x.stem
                )
            else:
                st.error("No datasets found")
                return
        else:
            st.error("Data directory not found")
            return
        
        st.markdown("---")
        
        # Options
        show_predictions = st.checkbox("Show Predictions", value=True, disabled=(model is None))
        show_uncertainty = st.checkbox("Show Uncertainty", value=True, disabled=(model is None))
        num_samples = st.slider("Number of samples", 10, 100, 50)
    
    # Main content
    if selected_dataset:
        # Load ground truth data
        ue_x, ue_y, measurements = load_dataset(str(selected_dataset))
        
        if ue_x is not None and ue_y is not None:
            gt_positions = np.column_stack([ue_x, ue_y])
            
            # Show basic stats
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Samples", len(gt_positions))
            with col2:
                st.metric("X Range", f"{ue_x.min():.0f} to {ue_x.max():.0f}m")
            with col3:
                st.metric("Y Range", f"{ue_y.min():.0f} to {ue_y.max():.0f}m")
            with col4:
                area = (ue_x.max() - ue_x.min()) * (ue_y.max() - ue_y.min())
                st.metric("Area", f"{area/1e6:.1f} km²")
            
            st.markdown("---")
            
            # Generate predictions if model available
            predictions_data = None
            if model and show_predictions:
                with st.spinner("Generating predictions..."):
                    predictions_data = generate_predictions(model, device, str(selected_dataset), num_samples)
            
            # Create visualization
            if predictions_data:
                st.subheader("Position Comparison: Ground Truth vs Predictions")
                
                # Calculate errors
                errors = np.linalg.norm(
                    predictions_data['gt_positions'] - predictions_data['predictions'], 
                    axis=1
                )
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Error", f"{np.median(errors):.1f}m")
                with col2:
                    st.metric("Mean Error", f"{np.mean(errors):.1f}m")
                with col3:
                    st.metric("90th %ile", f"{np.percentile(errors, 90):.1f}m")
                with col4:
                    st.metric("Max Error", f"{np.max(errors):.1f}m")
                
                # Main map
                fig = create_position_map(
                    predictions_data['gt_positions'],
                    predictions_data['predictions'],
                    predictions_data['uncertainties'] if show_uncertainty else None,
                    title="Ground Truth (green) vs Predictions (red X) - Lines show errors"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Error analysis
                st.subheader("Error Analysis")
                create_error_distribution(predictions_data['gt_positions'], predictions_data['predictions'])
                
            else:
                # Just show ground truth
                st.subheader("Ground Truth Positions")
                fig = create_position_map(gt_positions, title="UE Trajectory Ground Truth")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Failed to load dataset")


if __name__ == "__main__":
    main()
