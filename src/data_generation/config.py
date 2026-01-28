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
        allow_mock_fallback: If True, allows falling back to mock on Sionna failures.
        require_sionna: If True, fail fast when Sionna is unavailable or fails.
        require_cfr: If True, require CFR features to be present.
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
        cfr_num_subcarriers: Number of subcarriers to compute/store for CFR/CSI.
        measurement_dropout_rates: Dictionary of dropout rates for measurements.
        measurement_dropout_seed: Seed for measurement dropout randomness.
        quantization_enabled: Enable 3GPP quantization.
        output_dir: Output directory for the dataset.
    """
    scene_dir: Path
    scene_metadata_path: Path
    carrier_frequency_hz: float
    bandwidth_hz: float
    tx_power_dbm: float
    noise_figure_db: float
    use_mock_mode: bool = False
    allow_mock_fallback: bool = True
    require_sionna: bool = False
    require_cfr: bool = False
    max_depth: int = 5
    num_samples: int = 100_000
    enable_diffraction: bool = True
    enable_diffuse_reflection: bool = False
    enable_edge_diffraction: bool = False
    rt_batch_size: int = 16
    num_ue_per_tile: int = 100
    ue_height_range: Tuple[float, float] = (1.5, 1.5)
    ue_velocity_range: Tuple[float, float] = (0.0, 1.5)
    num_reports_per_ue: int = 10
    report_interval_ms: float = 200.0
    enable_k_factor: bool = False
    enable_beam_management: bool = True
    num_beams: int = 64
    max_neighbors: int = 8
    cfr_num_subcarriers: int = 64
    measurement_dropout_rates: Optional[Dict[str, float]] = None
    measurement_dropout_seed: int = 42
    quantization_enabled: bool = True
    output_dir: Path = Path('data/synthetic')
    max_stored_paths: int = 256
    max_stored_sites: int = 16
    enforce_unique_ue_positions: bool = False
    min_ue_separation_m: float = 1.0
    ue_sampling_margin_m: float = 0.0
    rt_diagnostics_max: int = 10
    rt_fail_log_every: int = 100
    drop_log_every: int = 100
    drop_failed_reports: bool = True
    max_resample_attempts: int = 10
    
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
            allow_mock_fallback=dg_config.get('allow_mock_fallback', True),
            require_sionna=dg_config.get('require_sionna', False),
            require_cfr=dg_config.get('require_cfr', False),
            max_depth=dg_config.get('max_depth', 5),
            num_samples=dg_config.get('num_samples', 1_000_000),
            enable_diffraction=dg_config.get('enable_diffraction', True),
            enable_diffuse_reflection=dg_config.get('enable_diffuse_reflection', False),
            enable_edge_diffraction=dg_config.get('enable_edge_diffraction', False),
            rt_batch_size=dg_config.get('rt_batch_size', 16),
            num_ue_per_tile=dg_config.get('num_ue_per_tile', 100),
            ue_height_range=tuple(dg_config.get('ue_height_range', [1.5, 1.5])),
            ue_velocity_range=tuple(dg_config.get('ue_velocity_range', [0.0, 1.5])),
            num_reports_per_ue=dg_config.get('num_reports_per_ue', 10),
            report_interval_ms=dg_config.get('report_interval_ms', 200.0),
            enable_k_factor=dg_config.get('enable_k_factor', False),
            enable_beam_management=dg_config.get('enable_beam_management', True),
            num_beams=dg_config.get('num_beams', 64),
            max_neighbors=dg_config.get('max_neighbors', 8),
            cfr_num_subcarriers=dg_config.get('cfr_num_subcarriers', 64),
            measurement_dropout_rates=dg_config.get('measurement_dropout_rates'),
            measurement_dropout_seed=dg_config.get('measurement_dropout_seed', 42),
            quantization_enabled=dg_config.get('quantization_enabled', True),
            output_dir=output_dir,
            enforce_unique_ue_positions=dg_config.get('enforce_unique_ue_positions', False),
            min_ue_separation_m=dg_config.get('min_ue_separation_m', 1.0),
            ue_sampling_margin_m=dg_config.get('ue_sampling_margin_m', 0.0),
            rt_diagnostics_max=dg_config.get('rt_diagnostics_max', 10),
            rt_fail_log_every=dg_config.get('rt_fail_log_every', 100),
            drop_log_every=dg_config.get('drop_log_every', 100),
            drop_failed_reports=dg_config.get('drop_failed_reports', True),
            max_resample_attempts=dg_config.get('max_resample_attempts', 10),
        )
