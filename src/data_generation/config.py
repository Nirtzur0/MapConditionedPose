"""
Configuration for multi-layer data generation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import yaml

@dataclass
class DataGenerationConfig:
    """Configuration for multi-layer data generation.
    
    Attributes:
        scene_dir: Directory containing M1 scenes.
        scene_metadata_path: Path to scene metadata.
        carrier_frequency_hz: Carrier frequency in Hz.
        bandwidth_hz: System bandwidth in Hz.
        tx_power_dbm: Transmit power in dBm.
        noise_figure_db: Receiver noise figure in dB.
        use_mock_mode: If True, uses mock data instead of Sionna.
        max_depth: Max reflections/diffractions for ray tracing.
        num_samples: Number of samples per source for path tracing.
        enable_diffraction: Enable diffraction in ray tracing.
        num_ue_per_tile: Number of UEs to sample per scene tile.
        ue_height_range: Min/max UE height in meters.
        ue_velocity_range: Min/max UE velocity in m/s.
        num_reports_per_ue: Number of measurement reports per UE trajectory.
        report_interval_ms: Time between reports in milliseconds.
        enable_k_factor: Compute Rician K-factor.
        enable_beam_management: Enable 5G NR beam management.
        num_beams: Number of SSB beams.
        max_neighbors: Max neighbor cells to report.
        measurement_dropout_rates: Dictionary of dropout rates for measurements.
        quantization_enabled: Enable 3GPP quantization.
        output_dir: Output directory for the Zarr dataset.
        zarr_chunk_size: Zarr chunk size (samples per chunk).
    """
    scene_dir: Path
    scene_metadata_path: Path
    carrier_frequency_hz: float
    bandwidth_hz: float
    tx_power_dbm: float
    noise_figure_db: float
    use_mock_mode: bool = False
    max_depth: int = 5
    num_samples: int = 100_000
    enable_diffraction: bool = True
    num_ue_per_tile: int = 100
    ue_height_range: Tuple[float, float] = (1.5, 1.5)
    ue_velocity_range: Tuple[float, float] = (0.0, 1.5)
    num_reports_per_ue: int = 10
    report_interval_ms: float = 200.0
    enable_k_factor: bool = False
    enable_beam_management: bool = True
    num_beams: int = 64
    max_neighbors: int = 8
    measurement_dropout_rates: Optional[Dict[str, float]] = None
    quantization_enabled: bool = True
    output_dir: Path = Path('data/synthetic')
    zarr_chunk_size: int = 100
    max_stored_paths: int = 256
    max_stored_sites: int = 16
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'DataGenerationConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract data_generation section
        dg_config = data.get('data_generation', {})
        
        # Convert paths
        scene_dir = Path(dg_config.get('scenes', {}).get('root_dir', 'data/scenes'))
        scene_metadata_path = Path(dg_config.get('scene_metadata_path', scene_dir / 'metadata.json'))
        output_dir = Path(dg_config.get('output', {}).get('path', 'data/processed/sionna_dataset'))
        
        return cls(
            scene_dir=scene_dir,
            scene_metadata_path=scene_metadata_path,
            carrier_frequency_hz=float(dg_config.get('carrier_frequency_hz', 3.5e9)),
            bandwidth_hz=float(dg_config.get('bandwidth_hz', 100e6)),
            tx_power_dbm=float(dg_config.get('tx_power_dbm', 43.0)),
            noise_figure_db=float(dg_config.get('noise_figure_db', 9.0)),
            use_mock_mode=dg_config.get('use_mock_mode', False),
            max_depth=dg_config.get('max_depth', 5),
            num_samples=dg_config.get('num_samples', 1_000_000),
            enable_diffraction=dg_config.get('enable_diffraction', True),
            num_ue_per_tile=dg_config.get('num_ue_per_tile', 100),
            ue_height_range=tuple(dg_config.get('ue_height_range', [1.5, 1.5])),
            ue_velocity_range=tuple(dg_config.get('ue_velocity_range', [0.0, 1.5])),
            num_reports_per_ue=dg_config.get('num_reports_per_ue', 10),
            report_interval_ms=dg_config.get('report_interval_ms', 200.0),
            enable_k_factor=dg_config.get('enable_k_factor', False),
            enable_beam_management=dg_config.get('enable_beam_management', True),
            num_beams=dg_config.get('num_beams', 64),
            max_neighbors=dg_config.get('max_neighbors', 8),
            measurement_dropout_rates=dg_config.get('measurement_dropout_rates'),
            quantization_enabled=dg_config.get('quantization_enabled', True),
            output_dir=output_dir,
            zarr_chunk_size=dg_config.get('zarr_chunk_size', 100),
        )
