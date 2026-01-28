"""
Unified pipeline configuration schema and loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from omegaconf import DictConfig, OmegaConf, MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    output_dir: str = MISSING
    clean: bool = False


@dataclass
class PipelineFlags:
    skip_scenes: bool = False
    skip_data: bool = False
    skip_training: bool = False


@dataclass
class CityConfig:
    name: str = MISSING
    bbox: List[float] = field(default_factory=list)
    split: str = "train"


@dataclass
class ScenesConfig:
    cities: List[CityConfig] = field(default_factory=list)
    num_tx: int = MISSING
    tx_variations: int = MISSING
    site_strategy: str = MISSING
    tx_height_range_m: Optional[Tuple[float, float]] = None
    bbox_scale: float = 1.0
    use_lidar: bool = True
    use_dem: bool = False
    hag_tiff_path: Optional[str] = None


@dataclass
class DataGenerationConfig:
    carrier_frequency_hz: float = MISSING
    bandwidth_hz: float = MISSING
    tx_power_dbm: float = 43.0
    noise_figure_db: float = 9.0
    use_mock_mode: bool = False
    allow_mock_fallback: bool = True
    require_sionna: bool = False
    require_cfr: bool = False
    max_depth: int = 5
    num_samples: int = 1000000
    enable_diffraction: bool = True
    enable_diffuse_reflection: bool = False
    enable_edge_diffraction: bool = False
    rt_batch_size: int = 16
    num_ue_per_tile: int = MISSING
    ue_height_range: Tuple[float, float] = (1.5, 1.5)
    ue_velocity_range: Tuple[float, float] = (0.0, 1.5)
    num_reports_per_ue: int = MISSING
    report_interval_ms: float = 200.0
    enable_k_factor: bool = False
    enable_beam_management: bool = True
    num_beams: int = 64
    max_neighbors: int = 8
    cfr_num_subcarriers: int = 64
    measurement_dropout_rates: Dict[str, float] = field(default_factory=dict)
    measurement_dropout_seed: int = 42
    quantization_enabled: bool = True
    max_stored_paths: int = 256
    max_stored_sites: int = 16
    enforce_unique_ue_positions: bool = False
    min_ue_separation_m: float = 1.0
    ue_sampling_margin_m: float = 0.0
    max_attempts_per_ue: int = 25
    drop_failed_ue_trajectories: bool = False
    rt_diagnostics_max: int = 10
    rt_fail_log_every: int = 100
    drop_log_every: int = 100
    drop_failed_reports: bool = True
    max_resample_attempts: int = 10
    min_scene_survival_ratio: float = 0.6
    max_scene_resample_attempts: int = 3
    split_mode: str = "scene"
    split_train_val_label: str = "train_val"
    split_ratios: Dict[str, float] = field(default_factory=dict)
    use_sionna_sys: bool = False
    num_allocated_re: int = 0
    bler_target: float = 0.1
    mcs_table_index: int = 1
    mcs_category: int = 0
    slot_duration_ms: float = 1.0
    log_fallback_warnings: bool = False
    kfold_num_folds: int = 5
    kfold_fold_index: int = 0
    kfold_shuffle: bool = True
    kfold_seed: int = 42


@dataclass
class RadioEncoderConfig:
    num_cells: int = MISSING
    num_beams: int = MISSING
    d_model: int = MISSING
    nhead: int = MISSING
    num_layers: int = MISSING
    dropout: float = MISSING
    max_seq_len: int = MISSING
    rt_features_dim: int = MISSING
    phy_features_dim: int = MISSING
    mac_features_dim: int = MISSING


@dataclass
class MapEncoderConfig:
    img_size: int = MISSING
    patch_size: int = MISSING
    in_channels: int = MISSING
    d_model: int = MISSING
    nhead: int = MISSING
    num_layers: int = MISSING
    dropout: float = MISSING
    radio_map_channels: int = MISSING
    osm_map_channels: int = MISSING
    cache_size: int = 0
    cache_mode: str = "off"


@dataclass
class FusionConfig:
    d_fusion: int = MISSING
    nhead: int = MISSING
    dropout: float = MISSING
    num_query_tokens: int = 4


@dataclass
class CoarseHeadConfig:
    grid_size: int = MISSING
    d_input: int = MISSING
    dropout: float = MISSING


@dataclass
class FineHeadConfig:
    d_input: int = MISSING
    d_hidden: int = MISSING
    top_k: int = MISSING
    patch_size: int = 64
    use_local_map: bool = False
    offset_scale: float = 1.5
    dropout: float = MISSING


@dataclass
class ModelConfig:
    radio_encoder: RadioEncoderConfig = field(default_factory=RadioEncoderConfig)
    map_encoder: MapEncoderConfig = field(default_factory=MapEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    coarse_head: CoarseHeadConfig = field(default_factory=CoarseHeadConfig)
    fine_head: FineHeadConfig = field(default_factory=FineHeadConfig)


@dataclass
class AuxiliaryLossConfig:
    enabled: bool = False
    weight: float = 0.1
    hidden_dim: int = 0
    input: str = "radio"
    tasks: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingLossConfig:
    coarse_weight: float = MISSING
    fine_weight: float = MISSING
    use_physics_loss: bool = False
    auxiliary: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)


@dataclass
class TrainingAugmentationConfig:
    feature_noise: float = 0.0
    feature_dropout: float = 0.0
    temporal_dropout: float = 0.0
    random_flip: bool = False
    random_rotation: bool = False
    scale_range: Tuple[float, float] = (1.0, 1.0)


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    num_epochs: int = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    warmup_steps: int = 0
    loss: TrainingLossConfig = field(default_factory=TrainingLossConfig)
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    dropout: float = 0.1
    gradient_clip: float = 1.0
    gradient_clip_val: float = 1.0
    augmentation: TrainingAugmentationConfig = field(default_factory=TrainingAugmentationConfig)


@dataclass
class DatasetConfig:
    train_lmdb_paths: List[str] = field(default_factory=list)
    val_lmdb_paths: List[str] = field(default_factory=list)
    test_lmdb_paths: List[str] = field(default_factory=list)
    require_lmdb: bool = False
    map_resolution: float = MISSING
    scene_extent: Any = MISSING
    normalize_features: bool = True
    normalize_maps: bool = False
    map_norm_mode: str = "zscore"
    map_log_throughput: bool = False
    map_log_epsilon: float = 1e-3
    handle_missing_values: str = "mask"
    split_seed: int = 42
    map_cache_size: int = 0
    sequence_length: int = 0
    max_cells: int = 2


@dataclass
class RefinementConfig:
    enabled: bool = False
    num_steps: int = 10
    learning_rate: float = 0.1
    min_confidence_threshold: float = 0.6
    clip_to_extent: bool = True
    coarse_logit_temperature: float = 1.0
    candidate_sigma_ratio: float = 0.05
    confidence_combine: str = "min"


@dataclass
class PhysicsLossConfig:
    enabled: bool = False
    lambda_phys: float = 0.1
    feature_weights: Dict[str, float] = field(default_factory=dict)
    loss_type: str = "mse"
    huber_delta: float = 1.0
    normalize_features: bool = True
    radio_maps_dir: str = "data/radio_maps"
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=list)
    error_bins: List[float] = field(default_factory=list)
    viz_frequency: int = 5


@dataclass
class CheckpointConfig:
    dirpath: str = "checkpoints"
    save_top_k: int = 3
    monitor: str = "val_median_error"
    mode: str = "min"


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    monitor: str = "val_median_error"
    mode: str = "min"


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    use_comet: bool = False
    project: str = "ue-localization"
    log_every_n_steps: int = 50


@dataclass
class InfrastructureConfig:
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    num_workers: int = 0
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class ProfilingConfig:
    enabled: bool = False
    output_dir: Optional[str] = None
    sort: str = "cumtime"
    top_n: int = 50
    save_raw: bool = True


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    pipeline: PipelineFlags = field(default_factory=PipelineFlags)
    scenes: ScenesConfig = field(default_factory=ScenesConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    physics_loss: PhysicsLossConfig = field(default_factory=PhysicsLossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    seed: int = 42
    deterministic: bool = False


def _validate_config(cfg: DictConfig) -> None:
    split_ratios = cfg.data_generation.split_ratios
    if split_ratios:
        total = sum(split_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {total}")

    for city in cfg.scenes.cities:
        if len(city.bbox) != 4:
            raise ValueError(f"City bbox must have 4 values: {city.name}")


def load_pipeline_config(path: Optional[Path]) -> DictConfig:
    schema = OmegaConf.structured(PipelineConfig)
    # OmegaConf.set_struct(schema, True)
    cfg = schema
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    OmegaConf.set_struct(cfg, True)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _validate_config(cfg)
    return cfg


def apply_quick_test_overrides(cfg: DictConfig) -> DictConfig:
    overrides = {
        "experiment": {"name": "quick_test"},
        "scenes": {
            "cities": [
                {
                    "name": "Boulder, CO",
                    "bbox": [-105.275, 40.016, -105.272, 40.018],
                    "split": "train",
                }
            ],
            "num_tx": 2,
            "tx_variations": 5,
            "site_strategy": "random",
        },
        "data_generation": {
            "num_ue_per_tile": 20,
            "num_reports_per_ue": 5,
            "split_ratios": {
                "train": 0.6,
                "val": 0.2,
                "test": 0.2
            }
        },
        "dataset": {
            "sequence_length": 5,
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.0002,
        },
    }
    cfg = OmegaConf.merge(cfg, overrides)
    _validate_config(cfg)
    return cfg


def apply_robust_overrides(cfg: DictConfig) -> DictConfig:
    overrides = {
        "experiment": {"name": "robust_training"},
        "scenes": {
            "cities": [
                {"name": "Boulder, CO", "bbox": [-105.280, 40.014, -105.270, 40.022], "split": "train"},
                {"name": "Austin, TX", "bbox": [-97.745, 30.265, -97.735, 30.275], "split": "train"},
                {"name": "Seattle, WA", "bbox": [-122.340, 47.605, -122.330, 47.615], "split": "train"},
                {"name": "Denver, CO", "bbox": [-104.995, 39.745, -104.985, 39.755], "split": "train"},
                {"name": "Portland, OR", "bbox": [-122.680, 45.520, -122.670, 45.530], "split": "train"},
                {"name": "Boston, MA", "bbox": [-71.060, 42.355, -71.050, 42.365], "split": "train"},
                {"name": "Chicago, IL", "bbox": [-87.630, 41.880, -87.620, 41.890], "split": "val"},
                {"name": "Phoenix, AZ", "bbox": [-112.075, 33.450, -112.065, 33.460], "split": "val"},
                {"name": "NYC, NY", "bbox": [-73.990, 40.750, -73.980, 40.760], "split": "test"},
                {"name": "Atlanta, GA", "bbox": [-84.390, 33.755, -84.380, 33.765], "split": "test"},
            ],
            "num_tx": 2,
            "tx_variations": 5,
            "site_strategy": "random",
        },
        "data_generation": {
            "num_ue_per_tile": 150,
            "num_reports_per_ue": 10,
        },
        "training": {
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0003,
        },
    }
    cfg = OmegaConf.merge(cfg, overrides)
    _validate_config(cfg)
    return cfg
