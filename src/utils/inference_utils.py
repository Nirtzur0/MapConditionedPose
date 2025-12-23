"""Utility functions for model inference and data preprocessing."""

import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


def load_model(model_path: Path, device: torch.device):
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def preprocess_measurements(measurements: List[Any]) -> torch.Tensor:
    """Preprocess measurement sequence for model input.
    
    Args:
        measurements: List of Measurement objects
        
    Returns:
        Tensor of shape [seq_len, num_features]
    """
    features_list = []
    
    for meas in measurements:
        # Extract all available features
        features = []
        
        # Metadata
        features.append(meas.timestamp)
        features.append(float(meas.cell_id))
        features.append(float(meas.beam_id) if meas.beam_id is not None else -1.0)
        
        # RT layer features (7 features)
        features.append(meas.path_gain if meas.path_gain is not None else -160.0)
        features.append(meas.toa if meas.toa is not None else 0.0)
        features.append(meas.aoa_azimuth if meas.aoa_azimuth is not None else 0.0)
        features.append(meas.aoa_zenith if meas.aoa_zenith is not None else 0.0)
        features.append(meas.doppler if meas.doppler is not None else 0.0)
        
        # PHY/FAPI layer features (6+ features)
        features.append(meas.rsrp if meas.rsrp is not None else -160.0)
        features.append(meas.rsrq if meas.rsrq is not None else -40.0)
        features.append(meas.sinr if meas.sinr is not None else -20.0)
        features.append(float(meas.cqi) if meas.cqi is not None else 0.0)
        features.append(float(meas.ri) if meas.ri is not None else 1.0)
        
        # MAC/RRC layer features (2+ features)
        features.append(float(meas.timing_advance) if meas.timing_advance is not None else 0.0)
        features.append(meas.phr if meas.phr is not None else 0.0)
        
        features_list.append(features)
    
    # Convert to tensor
    tensor = torch.tensor(features_list, dtype=torch.float32)
    
    return tensor


def create_dummy_measurement_sequence(
    num_steps: int = 10,
    scene_id: str = "demo_scene"
) -> List[Dict[str, Any]]:
    """Create dummy measurement sequence for testing.
    
    Args:
        num_steps: Number of time steps
        scene_id: Scene identifier
        
    Returns:
        List of measurement dictionaries
    """
    measurements = []
    
    for i in range(num_steps):
        meas = {
            "timestamp": i * 0.2,  # 200ms intervals
            "cell_id": np.random.choice([101, 105, 108]),
            "beam_id": np.random.randint(0, 64),
            
            # RT layer
            "path_gain": -80.0 + np.random.randn() * 5.0,
            "toa": np.random.uniform(0.1, 5.0) * 1e-6,  # microseconds
            "aoa_azimuth": np.random.uniform(0, 360),
            "aoa_zenith": np.random.uniform(0, 90),
            "doppler": np.random.randn() * 100,  # Hz
            
            # PHY/FAPI layer
            "rsrp": -85.0 + np.random.randn() * 8.0,
            "rsrq": -12.0 + np.random.randn() * 2.0,
            "sinr": 15.0 + np.random.randn() * 5.0,
            "cqi": np.random.randint(1, 15),
            "ri": np.random.randint(1, 4),
            
            # MAC/RRC layer
            "timing_advance": np.random.randint(0, 1000),
            "phr": np.random.uniform(0, 40),
        }
        measurements.append(meas)
    
    return measurements


def compute_error_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute positioning error metrics.
    
    Args:
        predictions: Array of shape [N, 2] with predicted positions
        ground_truth: Array of shape [N, 2] with true positions
        
    Returns:
        Dictionary of metrics
    """
    # Compute Euclidean errors
    errors = np.linalg.norm(predictions - ground_truth, axis=1)
    
    metrics = {
        "median_error_m": float(np.median(errors)),
        "percentile_67_m": float(np.percentile(errors, 67)),
        "percentile_90_m": float(np.percentile(errors, 90)),
        "percentile_95_m": float(np.percentile(errors, 95)),
        "mean_error_m": float(np.mean(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "success_rate_5m": float(np.mean(errors <= 5.0)),
        "success_rate_10m": float(np.mean(errors <= 10.0)),
    }
    
    return metrics


def normalize_coordinates(xy: torch.Tensor, map_extent: tuple) -> torch.Tensor:
    """Normalize coordinates to [-1, 1] for grid_sample.
    
    Args:
        xy: Coordinates of shape [..., 2] in meters
        map_extent: (min_x, min_y, max_x, max_y) in meters
        
    Returns:
        Normalized coordinates in [-1, 1]
    """
    min_x, min_y, max_x, max_y = map_extent
    
    x_norm = 2 * (xy[..., 0] - min_x) / (max_x - min_x) - 1
    y_norm = 2 * (xy[..., 1] - min_y) / (max_y - min_y) - 1
    
    return torch.stack([x_norm, y_norm], dim=-1)
