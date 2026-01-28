"""
Orchestrates multi-layer radio propagation simulation, feature extraction, and dataset generation.

This module has been refactored to use specialized classes for better maintainability:
- simulation.SceneLoader: Scene loading and metadata management
- simulation.TransmitterSetup: Transmitter/receiver configuration
- simulation.BatchSimulator: Ray tracing and feature extraction
- processing.DataStacker: Data aggregation and padding
- processing.MeasurementProcessor: Measurement realism
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import traceback

from .config import DataGenerationConfig
from .trajectory import sample_ue_trajectories
from .visualization import render_scene_3d, save_map_visualizations

from .features import (
    RTFeatureExtractor, PHYFAPIFeatureExtractor, MACRRCFeatureExtractor,
    RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures,
    SionnaNativeKPIExtractor
)
from .radio_map_generator import RadioMapGenerator, RadioMapConfig

# Import new specialized classes
from .simulation import SceneLoader, TransmitterSetup, BatchSimulator
from .processing import DataStacker, MeasurementProcessor

logger = logging.getLogger(__name__)

# Import dataset writer and OSM rasterizer
try:
    from .lmdb_writer import LMDBDatasetWriter
    from .osm_rasterizer import OSMRasterizer
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    logger.error("LMDB not available; install with: pip install lmdb")


# Try importing Sionna
try:
    import sionna
    from sionna.rt import load_scene
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available; MultiLayerDataGenerator will operate in mock mode.")

# Try importing tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from .radio_map_generator import RadioMapGenerator, RadioMapConfig
from .osm_rasterizer import OSMRasterizer
from .visualization import save_map_visualizations, render_scene_3d


class MultiLayerDataGenerator:
    """
    Orchestrates end-to-end data generation pipeline.
    
    This class has been refactored to use composition and dependency injection
    for better maintainability and testability. It delegates complex operations
    to specialized classes while maintaining the same public interface.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initializes the data generator with a configuration.
        
        Args:
            config: Data generation configuration
        """
        self.config = config

        if config.require_sionna and config.use_mock_mode:
            raise ValueError("use_mock_mode cannot be true when require_sionna is enabled.")
        if config.use_mock_mode and not config.allow_mock_fallback:
            raise ValueError("use_mock_mode requires allow_mock_fallback to be true.")
        if not SIONNA_AVAILABLE and (config.require_sionna or (not config.allow_mock_fallback and not config.use_mock_mode)):
            raise ImportError("Sionna is required but not available. Install sionna or disable strict mode.")
        
        # Initialize specialized components
        self.scene_loader = SceneLoader(
            scene_dir=config.scene_dir,
            carrier_frequency_hz=config.carrier_frequency_hz
        )
        
        self.transmitter_setup = TransmitterSetup(
            carrier_frequency_hz=config.carrier_frequency_hz
        )
        
        # Initialize feature extractors
        self.rt_extractor = RTFeatureExtractor(
            carrier_frequency_hz=config.carrier_frequency_hz,
            bandwidth_hz=config.bandwidth_hz,
            compute_k_factor=config.enable_k_factor,
            max_stored_paths=config.max_stored_paths,
            max_stored_sites=config.max_stored_sites,
            allow_mock_fallback=config.allow_mock_fallback,
        )
        
        self.phy_extractor = PHYFAPIFeatureExtractor(
            noise_figure_db=config.noise_figure_db,
            enable_beam_management=config.enable_beam_management,
            num_beams=config.num_beams,
        )
        
        self.mac_extractor = MACRRCFeatureExtractor(
            max_neighbors=config.max_neighbors,
            enable_throughput=True,
            enable_handover=False,
            use_sionna_sys=getattr(config, "use_sionna_sys", False),
            num_allocated_re=getattr(config, "num_allocated_re", 0),
            bler_target=getattr(config, "bler_target", 0.1),
            mcs_table_index=getattr(config, "mcs_table_index", 1),
            mcs_category=getattr(config, "mcs_category", 0),
            slot_duration_ms=getattr(config, "slot_duration_ms", 1.0),
        )

        self.native_extractor = SionnaNativeKPIExtractor(
            carrier_frequency_hz=config.carrier_frequency_hz,
            bandwidth_hz=config.bandwidth_hz,
            noise_figure_db=config.noise_figure_db,
            max_paths=config.max_stored_paths
        )
        
        # Initialize Sionna Backbone (OFDM/MIMO)
        self.rg = None
        if SIONNA_AVAILABLE:
            self.rg = self._initialize_resource_grid()
        
        # Initialize batch simulator
        self.batch_simulator = BatchSimulator(
            rt_extractor=self.rt_extractor,
            phy_extractor=self.phy_extractor,
            mac_extractor=self.mac_extractor,
            native_extractor=self.native_extractor,
            config=config,
            resource_grid=self.rg
        )
        
        # Initialize processing components
        self.data_stacker = DataStacker(
            max_neighbors=config.max_neighbors,
            num_subcarriers=getattr(config, 'cfr_num_subcarriers', 64),
        )
        self.measurement_processor = MeasurementProcessor(config=config)
        
        # Initialize LMDB writer (required)
        if not LMDB_AVAILABLE:
            raise ImportError("LMDB not available; install with: pip install lmdb")

        logger.info("Using LMDB for dataset storage (multiprocessing-friendly)")
        self.lmdb_writer = LMDBDatasetWriter(
            output_dir=config.output_dir,
            map_size=config.lmdb_map_size if hasattr(config, 'lmdb_map_size') else 100 * 1024**3,
            sequence_length=config.num_reports_per_ue,
        )
        # Set max dimensions
        self.lmdb_writer.set_max_dimensions({
            'max_cells': config.max_neighbors,
            'max_paths': config.max_stored_paths,
            'max_beams': 64,  # Default
        })
        
        # Initialize Map Generators
        self.radio_map_generator = None
        self.osm_rasterizer = None

        if SIONNA_AVAILABLE:
            rm_config = RadioMapConfig(
                resolution=1.0,
                map_size=(256, 256),
                map_extent=(0, 0, 512, 512), # Will be updated per scene
                ue_height=1.5,
                carrier_frequency=config.carrier_frequency_hz,
                bandwidth=config.bandwidth_hz,
                tx_power=config.tx_power_dbm,
                noise_figure=config.noise_figure_db,
                output_dir=config.output_dir / "radio_maps"
            )
            self.radio_map_generator = RadioMapGenerator(rm_config)
            self.osm_rasterizer = OSMRasterizer(map_size=(256, 256)) # Extent updated later
        
        logger.info(f"DataGenerator initialized for scene directory: {config.scene_dir}")
        logger.info(f"  Carrier freq: {config.carrier_frequency_hz/1e9:.2f} GHz")
        logger.info(f"  UEs per tile: {config.num_ue_per_tile}")
        logger.info(f"  Reports per UE: {config.num_reports_per_ue}")
        logger.info(f"  Sionna RT: {'enabled' if SIONNA_AVAILABLE else 'disabled'}")
    
    def _initialize_resource_grid(self) -> Optional[Any]:
        """Initialize Sionna resource grid for OFDM."""
        try:
            from sionna.phy.ofdm import ResourceGrid
            
            scs = 30e3
            num_subcarriers = getattr(self.config, 'cfr_num_subcarriers', None)
            if num_subcarriers is None:
                num_subcarriers = getattr(self.config, 'num_subcarriers', 1024)
            fft_size = 2**int(np.ceil(np.log2(num_subcarriers)))
            
            rg = ResourceGrid(
                num_ofdm_symbols=14,
                fft_size=fft_size,
                subcarrier_spacing=scs,
                num_tx=1,
                num_streams_per_tx=1,
                cyclic_prefix_length=20,
                pilot_pattern=None,
                pilot_ofdm_symbol_indices=None
            )
            logger.info(f"Sionna Backbone initialized: OFDM Resource Grid ({fft_size} FFT, {scs/1e3} kHz SCS)")
            return rg
        except Exception as e:
            logger.warning(f"Failed to initialize Sionna Backbone: {e}")
            return None
    


    def compute_max_dimensions(self, scene_ids: List[str]) -> Dict[str, Tuple]:
        """
        Pre-compute maximum dimensions for variable-length features across all scenes.
        
        Args:
            scene_ids: List of scene IDs to scan.
            
        Returns:
            Dictionary of max dimensions (e.g., {'rt/path_gains': (2, 175)})
        """
        logger.info("Computing max dimensions across all scenes...")
        max_paths = 0
        max_neighbors = self.config.max_neighbors
        
        for scene_id in tqdm(scene_ids, desc="Scanning Metadata"):
            try:
                meta = self.scene_loader.load_metadata(scene_id)
                # Check for cached max paths in metadata
                if 'stats' in meta and 'max_paths' in meta['stats']:
                     m_p = meta['stats']['max_paths']
                     if m_p > max_paths: max_paths = m_p
                else:
                     # Fallback or estimation if not in metadata
                     # For now, use config default or safe upper bound if unknown
                     pass
            except Exception:
                pass
        
        # If metadata stats missing, use config limit or safe default
        if max_paths == 0:
            max_paths = self.config.max_stored_paths
            
        logger.info(f"Global Max Dimensions -> Paths: {max_paths}, Neighbors: {max_neighbors}")
        
        # Define shapes
        # RT Layer: [num_cells, max_paths]
        # For 'rt/path_gains': (num_cells, max_paths)
        # But wait, rt_extractor returns [num_cells, max_paths] arrays per sample?
        # Let's check RTFeatureExtractor.
        # It typically returns [num_rx, num_tx, max_paths] or flattened [num_cells, max_paths]
        # But wait, if we have multiple cells, is it [N, num_cells, max_paths]?
        # Let's check typical output. 
        # RTFeatureExtractor usually consolidates paths or returns per-link.
        # Assuming [N, num_cells, max_paths] for now based on 'rt/path_gains'.
        # 'rt/path_gains': np.random.randn(num_samples, max_paths)
        # It seems it treats it as [N, max_paths] -- implying paths are aggregated or single cell?
        # BUT `rt_extractor` in `BatchSimulator` usually iterates over cells.
        # In `features.py` (not shown), it often returns `(num_paths_total,)` or `(num_cells, max_paths)`.
        # Given the ambiguity, I'll set dimensions based on config.
        
        # So we return the tuple of dimensions *after* N.
        
        # We need to know the number of cells? 
        # Num cells varies per scene? 
        # If num_cells varies, we can't use a fixed 2nd dimension for all unless we pad to MAX_CELLS.
        # Variable length inner dimensions (cells) is tricky.
        # The fix plan says: "Fix Shape Mismatch issues".
        # If cells vary, we must pad to a global max_cells or store as variable length (object).
        # We will assume we want fixed array shapes for ML.
        
        # Scan for max cells too
        max_cells = 0
        for scene_id in scene_ids:
             try:
                meta = self.scene_loader.load_metadata(scene_id)
                num_sites = len([s for s in meta.get('sites', []) if s.get('cell_id')])
                if num_sites > max_cells: max_cells = num_sites
             except: pass
        
        if max_cells == 0: max_cells = 1 # Fallback
        
        # Let's assume [N, num_cells, max_paths] for RT if multi-cell.
        # Or [N, max_paths] if it picks best cell?
        # path_gains: [N, max_paths]  <-- looks like flattened or single cell?
        # path_gains: [N, max_paths]
        # rsrp: [N, num_cells]
        
        # I will set max dims for `rt/path_gains` to include max_cells if the data is 3D.
        # If the data is 2D (paths from all cells merged), then just max_paths.
        # `RTFeatureExtractor` usually pads to `max_stored_paths` per link, or total?
        # Use `self.config.max_stored_paths`.
        
        dims = {}
        # We need to verify what `RTFeatureExtractor` produces.
        # Assuming it produces [num_cells, max_paths_per_cell], then for N samples it is [N, num_cells, max_paths].
        # Let's be safe and set it to whatever `max_stored_paths` is configured to.
        # If the extractor produces [N, C, P], and we say max_dim is (C, P), that works.
        
        # For now, simple mapping based on config:
        
        # Default to at least 2 cells (mock sites) if none found
        max_cells = max(max_cells, 2)
        
        # RT Layer
        # Assuming shape (num_cells, max_stored_paths) for single sample
        dims['rt/path_gains'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_delays'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_aoa_azimuth'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_aoa_elevation'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_aod_azimuth'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_aod_elevation'] = (max_cells, self.config.max_stored_paths)
        dims['rt/path_doppler'] = (max_cells, self.config.max_stored_paths)
        
        # PHY Layer (add beam/path dimension)
        # Observed shape (N, 16, 16) -> (max_cells, max_beams)
        max_beams = 64 # Safe upper bound
        dims['phy_fapi/rsrp'] = (max_cells, max_beams)
        dims['phy_fapi/rsrq'] = (max_cells, max_beams)
        dims['phy_fapi/sinr'] = (max_cells, max_beams)
        dims['phy_fapi/cqi'] = (max_cells, 1) # CQI usually scalar per cell
        dims['phy_fapi/ri'] = (max_cells, 1)
        dims['phy_fapi/pmi'] = (max_cells, 1)
        
        # MAC Layer
        dims['mac_rrc/neighbor_cell_ids'] = (max_cells, max_neighbors)
        # serving_cell_id is scalar [N]
        # timing_advance is scalar [N]
        
        return dims

    def generate_dataset(self,
                        scene_ids: Optional[List[str]] = None,
                        num_scenes: Optional[int] = None,
                        create_splits: bool = True,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Dict[str, Path]:
        """
        Generates a complete dataset from M1 scenes.
        
        Args:
            scene_ids: Specific scene IDs to process (defaults to all).
            num_scenes: Limits the number of scenes processed (for testing).
            train_ratio: Proportion of data for training (default: 0.70)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            
        Returns:
            Dict with paths to generated datasets {'train': Path, 'val': Path, 'test': Path}
            or single Path if create_splits=False
        """
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Split ratios must sum to 1.0"
        # Discover scenes
        if scene_ids is None:
            # Find all scene.xml files recursively
            scene_xmls = sorted(list(self.config.scene_dir.rglob('scene.xml')))
            scene_ids = []
            for xml_path in scene_xmls:
                try:
                    # Use relative path from root scene_dir as scene_id
                    rel_path = xml_path.parent.relative_to(self.config.scene_dir)
                    scene_ids.append(str(rel_path))
                except ValueError:
                    continue
        
        if num_scenes:
            scene_ids = scene_ids[:num_scenes]
        
        logger.info(f"Processing {len(scene_ids)} scenes...")
        
        # Collect all data from scenes first
        all_data = []
        split_labels = []
        
        # Process each scene
        for scene_id in tqdm(scene_ids, desc="Generating Data from Scenes"):
            scene_path = self.config.scene_dir / scene_id / 'scene.xml'
            
            if not scene_path.exists():
                logger.warning(f"Scene file not found: {scene_path}")
                continue
            
            try:
                # Load metadata
                scene_metadata = self.scene_loader.load_metadata(scene_id)
                scene_metadata = dict(scene_metadata or {})
                split_label = self._load_scene_split(scene_path, scene_metadata)
                if split_label:
                    scene_metadata.setdefault("split", split_label)
                split_labels.append(scene_metadata.get("split"))
                scene_data = self.generate_scene_data(scene_path, scene_id, scene_metadata=scene_metadata)
                
                if scene_data:
                    all_data.append((scene_id, scene_data, scene_metadata))
                    
            except Exception as e:
                logger.error(f"Failed to process scene {scene_id}: {e}")
                logger.error(traceback.format_exc())
        
        if not all_data:
            logger.error("No data generated from any scene")
            return None
        
        logger.info(f"Generated data from {len(all_data)} scenes")
        
        # Create splits if requested
        if create_splits:
            if any(split_labels):
                split_mode = getattr(self.config, "split_mode", "scene")
                train_val_label = getattr(self.config, "split_train_val_label", "train_val")
                if split_mode == "train_val_kfold" or any(label == train_val_label for label in split_labels):
                    return self._create_trainval_kfold_datasets(all_data, train_val_label=train_val_label)
                return self._create_split_datasets_from_labels(all_data)
            return self._create_split_datasets(all_data, train_ratio, val_ratio, test_ratio)
        else:
            # Create single dataset
            writer = self.lmdb_writer

            if hasattr(writer, "set_split_ratios"):
                writer.set_split_ratios(
                    {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
                    split_seed=42,
                )
            for scene_id, scene_data, scene_metadata in tqdm(all_data, desc="Writing to database"):
                # Write scene maps first
                if 'radio_map' in scene_data and 'osm_map' in scene_data:
                    writer.write_scene_maps(
                        scene_id=scene_id,
                        radio_map=scene_data['radio_map'],
                        osm_map=scene_data['osm_map']
                    )

                # Append sample data
                writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)

            output_path = writer.finalize()
            logger.info(f"Dataset saved to: {output_path}")
            return output_path

    def _load_scene_split(self, scene_path: Path, scene_metadata: Optional[Dict]) -> Optional[str]:
        """Load split label from split.txt or metadata."""
        if scene_metadata and scene_metadata.get("split"):
            return scene_metadata.get("split")

        split_path = scene_path.parent / "split.txt"
        if split_path.exists():
            label = split_path.read_text().strip().lower()
            train_val_label = getattr(self.config, "split_train_val_label", "train_val")
            if label in {"train", "val", "test", train_val_label}:
                return label
        return None

    def _create_trainval_kfold_datasets(self, all_data, train_val_label: str = "train_val") -> Dict[str, Path]:
        """Create train/val splits via K-fold over train_val scenes, with test from explicit test scenes."""
        train_val_data, test_data = [], []
        for scene_id, scene_data, scene_metadata in all_data:
            split_label = (scene_metadata or {}).get("split")
            if split_label == "test":
                test_data.append((scene_id, scene_data, scene_metadata))
            elif split_label in {train_val_label, "train", "val", None}:
                train_val_data.append((scene_id, scene_data, scene_metadata))
            else:
                logger.warning("Scene %s has unknown split label '%s'; skipping.", scene_id, split_label)

        logger.info(
            "Split (train_val + kfold): Train/Val=%d scenes, Test=%d scenes",
            len(train_val_data),
            len(test_data),
        )

        output_paths: Dict[str, Path] = {}

        if train_val_data:
            writer = LMDBDatasetWriter(
                output_dir=self.config.output_dir,
                map_size=self.config.lmdb_map_size if hasattr(self.config, 'lmdb_map_size') else 100 * 1024**3,
                split_name="train_val",
                sequence_length=self.config.num_reports_per_ue,
            )
            writer.set_max_dimensions({
                'max_cells': self.config.max_neighbors,
                'max_paths': self.config.max_stored_paths,
                'max_beams': 64,
            })
            writer.set_kfold(
                num_folds=getattr(self.config, "kfold_num_folds", 5),
                fold_index=getattr(self.config, "kfold_fold_index", 0),
                shuffle=getattr(self.config, "kfold_shuffle", True),
                seed=getattr(self.config, "kfold_seed", 42),
            )
            for scene_id, scene_data, scene_metadata in tqdm(train_val_data, desc="Writing train_val"):
                if 'radio_map' in scene_data and 'osm_map' in scene_data:
                    writer.write_scene_maps(
                        scene_id=scene_id,
                        radio_map=scene_data['radio_map'],
                        osm_map=scene_data['osm_map']
                    )
                writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)
            train_val_path = writer.finalize()
            output_paths["train"] = train_val_path
            output_paths["val"] = train_val_path
            output_paths["train_val"] = train_val_path
            logger.info("✓ train/val (kfold): %s", train_val_path)
        else:
            logger.warning("No data for train_val pool; train/val splits will be empty.")

        if test_data:
            logger.info("Creating test dataset...")
            writer = LMDBDatasetWriter(
                output_dir=self.config.output_dir,
                map_size=self.config.lmdb_map_size if hasattr(self.config, 'lmdb_map_size') else 100 * 1024**3,
                split_name="test",
                sequence_length=self.config.num_reports_per_ue,
            )
            writer.set_max_dimensions({
                'max_cells': self.config.max_neighbors,
                'max_paths': self.config.max_stored_paths,
                'max_beams': 64,
            })
            for scene_id, scene_data, scene_metadata in tqdm(test_data, desc="Writing test"):
                if 'radio_map' in scene_data and 'osm_map' in scene_data:
                    writer.write_scene_maps(
                        scene_id=scene_id,
                        radio_map=scene_data['radio_map'],
                        osm_map=scene_data['osm_map']
                    )
                writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)
            output_path = writer.finalize()
            output_paths["test"] = output_path
            logger.info("✓ test: %s", output_path)
        else:
            logger.warning("No data for test split.")

        return output_paths

    def _create_split_datasets_from_labels(self, all_data) -> Dict[str, Path]:
        """Create train/val/test datasets based on explicit scene split labels."""
        train_data, val_data, test_data = [], [], []
        for scene_id, scene_data, scene_metadata in all_data:
            split_label = (scene_metadata or {}).get("split")
            if split_label == "train":
                train_data.append((scene_id, scene_data, scene_metadata))
            elif split_label == "val":
                val_data.append((scene_id, scene_data, scene_metadata))
            elif split_label == "test":
                test_data.append((scene_id, scene_data, scene_metadata))
            else:
                logger.warning("Scene %s missing split label; skipping.", scene_id)

        logger.info(
            "Split (explicit): Train=%d, Val=%d, Test=%d scenes",
            len(train_data),
            len(val_data),
            len(test_data),
        )

        output_paths = {}
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            if not split_data:
                logger.warning(f"No data for {split_name} split")
                continue
            logger.info(f"Creating {split_name} dataset...")
            writer = LMDBDatasetWriter(
                output_dir=self.config.output_dir,
                map_size=self.config.lmdb_map_size if hasattr(self.config, 'lmdb_map_size') else 100 * 1024**3,
                split_name=split_name,
                sequence_length=self.config.num_reports_per_ue,
            )
            writer.set_max_dimensions({
                'max_cells': self.config.max_neighbors,
                'max_paths': self.config.max_stored_paths,
                'max_beams': 64,
            })
            for scene_id, scene_data, scene_metadata in tqdm(split_data, desc=f"Writing {split_name}"):
                writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)
                if 'radio_map' in scene_data and 'osm_map' in scene_data:
                    writer.write_scene_maps(
                        scene_id=scene_id,
                        radio_map=scene_data['radio_map'],
                        osm_map=scene_data['osm_map']
                    )
            output_path = writer.finalize()
            output_paths[split_name] = output_path
            logger.info(f"✓ {split_name}: {output_path}")

        return output_paths
    
    def _create_split_datasets(self, all_data, train_ratio, val_ratio, test_ratio) -> Dict[str, Path]:
        """Create separate train/val/test LMDB datasets from collected data."""
        
        # Shuffle data for random split
        import random
        random.seed(42)  # Reproducible splits
        all_data_shuffled = all_data.copy()
        random.shuffle(all_data_shuffled)
        
        # Calculate split indices
        total = len(all_data_shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = all_data_shuffled[:train_end]
        val_data = all_data_shuffled[train_end:val_end]
        test_data = all_data_shuffled[val_end:]
        
        logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)} scenes")
        
        # Create writers for each split
        output_paths = {}
        
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if not split_data:
                logger.warning(f"No data for {split_name} split")
                continue
            
            logger.info(f"Creating {split_name} dataset...")
            
            # Create writer for this split (LMDB only)
            writer = LMDBDatasetWriter(
                output_dir=self.config.output_dir,
                map_size=self.config.lmdb_map_size if hasattr(self.config, 'lmdb_map_size') else 100 * 1024**3,
                split_name=split_name,
                sequence_length=self.config.num_reports_per_ue,
            )
            # Set max dimensions
            writer.set_max_dimensions({
                'max_cells': self.config.max_neighbors,
                'max_paths': self.config.max_stored_paths,
                'max_beams': 64,
            })
            
            # Write data
            for scene_id, scene_data, scene_metadata in tqdm(split_data, desc=f"Writing {split_name}"):
                writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)
                
                if 'radio_map' in scene_data and 'osm_map' in scene_data:
                    writer.write_scene_maps(
                        scene_id=scene_id,
                        radio_map=scene_data['radio_map'],
                        osm_map=scene_data['osm_map']
                    )
            
            # Finalize
            output_path = writer.finalize()
            output_paths[split_name] = output_path
            logger.info(f"✓ {split_name}: {output_path}")
        
        return output_paths

    def generate_scene_data(self,
                           scene_path: Path,
                           scene_id: str,
                           scene_metadata: Optional[Dict] = None,
                           scene_obj: Any = None) -> Dict[str, np.ndarray]:
        """
        Generates multi-layer data for a single scene.
        
        Args:
            scene_path: Path to scene.xml file
            scene_id: Scene identifier
            scene_metadata: Optional pre-loaded metadata
            scene_obj: Optional pre-loaded scene object
            
        Returns:
            Dictionary of stacked feature arrays
        """
        # Load scene-specific metadata
        if scene_metadata is None:
            scene_metadata = self.scene_loader.load_metadata(scene_id)

        def _sample_trajectories(relaxed: bool) -> List[np.ndarray]:
            return sample_ue_trajectories(
                scene_metadata=scene_metadata,
                num_ue_per_tile=self.config.num_ue_per_tile,
                ue_height_range=self.config.ue_height_range,
                ue_velocity_range=self.config.ue_velocity_range,
                num_reports_per_ue=self.config.num_reports_per_ue,
                report_interval_ms=self.config.report_interval_ms,
                offset=(0.0, 0.0),
                building_height_map=building_map,
                max_attempts_per_ue=max(25, getattr(self.config, "max_attempts_per_ue", 25)) if relaxed
                else getattr(self.config, "max_attempts_per_ue", 25),
                drop_failed_ue_trajectories=False if relaxed else getattr(self.config, "drop_failed_ue_trajectories", False),
                log_fallback_warnings=getattr(self.config, "log_fallback_warnings", False),
                enforce_unique_positions=False if relaxed else getattr(self.config, "enforce_unique_ue_positions", False),
                min_ue_separation_m=0.0 if relaxed else getattr(self.config, "min_ue_separation_m", 1.0),
                sampling_margin_m=0.0 if relaxed else getattr(self.config, "ue_sampling_margin_m", 0.0),
            )
        
        # Load scene in Sionna
        scene = scene_obj
        if scene is None:
            if SIONNA_AVAILABLE:
                scene = self.scene_loader.load_scene(scene_path)
            else:
                logger.warning("Sionna unavailable; using mock data.")
                scene = None
        
        # Extract site information
        site_positions, cell_ids, sites = self._extract_site_info(scene_metadata)
        
        # Calculate coordinate offsets
        sim_offset_x, sim_offset_y, store_offset_x, store_offset_y = self._calculate_offsets(scene_metadata)
        
        # Prepare site positions for simulation
        site_positions_sim = self._transform_to_sim_frame(site_positions, sim_offset_x, sim_offset_y)
        
        # Setup transmitters in Sionna
        if scene is not None:
            self.transmitter_setup.setup_transmitters(scene, site_positions_sim, sites if sites else {})
        
        # Sample UE trajectories
        building_map = None
        building_map_path = scene_path.parent / "2D_Building_Height_Map.npy"
        if building_map_path.exists():
            try:
                building_map = np.load(building_map_path)
            except Exception as e:
                logger.warning(f"Failed to load building height map from {building_map_path}: {e}")

        min_ratio = getattr(self.config, "min_scene_survival_ratio", 0.6)
        max_attempts = max(1, getattr(self.config, "max_scene_resample_attempts", 3))
        expected_reports = max(1, int(self.config.num_ue_per_tile) * int(self.config.num_reports_per_ue))

        for attempt in range(1, max_attempts + 1):
            relaxed = attempt > 1
            if relaxed:
                logger.warning(
                    "Resampling scene %s with relaxed UE constraints (attempt %d/%d).",
                    scene_id,
                    attempt,
                    max_attempts,
                )

            trajectories_global = _sample_trajectories(relaxed)
            if not trajectories_global:
                logger.warning("No UE trajectories for scene %s on attempt %d.", scene_id, attempt)
                continue

            ue_ratio = len(trajectories_global) / max(1, int(self.config.num_ue_per_tile))
            if ue_ratio < min_ratio:
                logger.warning(
                    "UE placement below threshold for scene %s (%.1f%% < %.1f%%).",
                    scene_id,
                    ue_ratio * 100.0,
                    min_ratio * 100.0,
                )
                continue

            # Process trajectories in batches
            all_features = self._process_trajectories_in_batches(
                trajectories_global,
                scene,
                site_positions_sim,
                cell_ids,
                sim_offset_x,
                sim_offset_y,
                store_offset_x,
                store_offset_y,
                scene_metadata=scene_metadata,
                building_map=building_map,
            )

            # Stack into arrays
            scene_data = self.data_stacker.stack_scene_data(all_features)
            if not scene_data:
                logger.warning("No stacked features for scene %s on attempt %d.", scene_id, attempt)
                continue

            actual_reports = sum(len(x) for x in all_features.get('positions', []) if x is not None)
            survival_ratio = actual_reports / expected_reports
            if survival_ratio < min_ratio:
                logger.warning(
                    "Scene %s survival ratio %.1f%% below threshold %.1f%%.",
                    scene_id,
                    survival_ratio * 100.0,
                    min_ratio * 100.0,
                )
                continue

            # Apply measurement realism
            scene_data = self.measurement_processor.apply_realism(scene_data)
            break
        else:
            logger.error("Skipping scene %s after %d failed resample attempts.", scene_id, max_attempts)
            return {}

        # Log statistics
        stats = self.batch_simulator.get_statistics()
        total = stats['sionna_ok'] + stats['sionna_fail']
        if total > 0:
            logger.info(
                "Sionna RT summary: %s/%s successful attempts, %s failed attempts (retried via batch splitting when possible)",
                stats["sionna_ok"],
                total,
                stats["sionna_fail"],
            )
        
        # --- Generate Maps ---
        if SIONNA_AVAILABLE and scene is not None:
            # Update configs based on actual scene bbox (if available) or config
            bbox = scene_metadata.get('bbox', {})
            x_min_glob = bbox.get('x_min', 0)
            y_min_glob = bbox.get('y_min', 0)
            x_max_glob = bbox.get('x_max', 512)
            y_max_glob = bbox.get('y_max', 512)
            
            # Simulation runs in a local frame centered at (0,0) geometry
            # The bbox width/height determines the extent
            width = x_max_glob - x_min_glob
            height = y_max_glob - y_min_glob
            
            # Center the map extent at 0,0
            x_min = -width / 2.0
            y_min = -height / 2.0
            x_max = width / 2.0
            y_max = height / 2.0
            
            # Update Radio Map Config: use fixed 256x256 for model compatibility
            self.radio_map_generator.config.map_extent = (x_min, y_min, x_max, y_max)
            self.radio_map_generator.config.map_size = (256, 256)
            
            # Generate Radio Map
            # Need cell sites from processing
            # We have 'sites' list from _extract_site_info
            
            # Re-extract sites as they might have been transformed? No, use original metadata sites.
            sites = scene_metadata.get('sites', [])
            
            logger.info(f"Generating Radio Map for {scene_id} (256x256)...")
            radio_map, radio_map_obj = self.radio_map_generator.generate_for_scene(
                scene, sites, return_sionna_object=True
            )
            
            # Model expects 5 channels: rsrp, rsrq, sinr, cqi, throughput
            # But we keep full map for visualization
            if radio_map.shape[0] > 5:
                radio_map_model = radio_map[:5]
            else:
                radio_map_model = radio_map
            
            scene_data['radio_map'] = radio_map_model
            
            # Visualization Setup
            viz_dir = self.config.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 3D Render (Radio Map)
            try:
                render_scene_3d(
                    scene=scene,
                    scene_id=scene_id,
                    metadata=scene_metadata,
                    output_dir=viz_dir / "3d",
                    radio_map=radio_map_obj
                )
            except Exception as e:
                logger.warning(f"3D Rendering failed: {e}")
            
            # Generate OSM Map
            # Update Rasterizer to use 256x256
            self.osm_rasterizer.width = 256
            self.osm_rasterizer.height = 256
            self.osm_rasterizer.x_min = x_min
            self.osm_rasterizer.y_min = y_min
            self.osm_rasterizer.x_max = x_max
            self.osm_rasterizer.y_max = y_max
            # Recompute scale/offset for 256x256 grid
            self.osm_rasterizer.resolution_x = (x_max - x_min) / 256
            self.osm_rasterizer.resolution_y = (y_max - y_min) / 256
            self.osm_rasterizer.scale = np.array([1.0 / self.osm_rasterizer.resolution_x, 1.0 / self.osm_rasterizer.resolution_y])
            self.osm_rasterizer.offset = np.array([-x_min, -y_min])
            
            logger.info(f"Generating OSM Map for {scene_id}...")
            osm_map = self.osm_rasterizer.rasterize(scene, scene_metadata=scene_metadata)
            scene_data['osm_map'] = osm_map

            # 2. 2D Map Plots (Radio + OSM)
            try:
                save_map_visualizations(
                    scene_id=scene_id,
                    radio_map=radio_map, # Pass full map for detailed viz
                    osm_map=osm_map,
                    output_dir=viz_dir
                )
            except Exception as e:
                logger.warning(f"2D Map Visualization failed: {e}")

        return scene_data
    
    def _extract_site_info(self, scene_metadata: Dict) -> Tuple[np.ndarray, np.ndarray, List]:
        """Extract site positions and cell IDs from metadata."""
        sites = scene_metadata.get('sites', [])
        valid_sites = [s for s in sites if s.get('cell_id') is not None]
        
        if len(valid_sites) == 0:
            logger.warning("No sites in metadata, using mock sites")
            bbox = scene_metadata.get('bbox', {})
            cx = (bbox.get('x_min', 0) + bbox.get('x_max', 0)) / 2
            cy = (bbox.get('y_min', 0) + bbox.get('y_max', 0)) / 2
            site_positions = np.array([[cx, cy, 30.0], [cx + 500.0, cy, 30.0]], dtype=np.float32)
            cell_ids = np.array([1, 2])
            return site_positions, cell_ids, []
        
        site_positions = np.array([s['position'] for s in valid_sites], dtype=np.float32)
        cell_ids = np.array([s['cell_id'] for s in valid_sites], dtype=int)
        return site_positions, cell_ids, sites
    
    def _calculate_offsets(self, scene_metadata: Dict) -> Tuple[float, float, float, float]:
        """Calculate coordinate frame offsets."""
        bbox = scene_metadata.get('bbox', {})
        
        if 'x_min' in bbox and 'x_max' in bbox:
            sim_offset_x = (bbox['x_min'] + bbox['x_max']) / 2
            sim_offset_y = (bbox['y_min'] + bbox['y_max']) / 2
            store_offset_x = bbox['x_min']
            store_offset_y = bbox['y_min']
            
            logger.info(f"Coord Frames -> Sim Offset: ({sim_offset_x:.2f}, {sim_offset_y:.2f}), "
                       f"Store Offset: ({store_offset_x:.2f}, {store_offset_y:.2f})")
        else:
            sim_offset_x, sim_offset_y = 0.0, 0.0
            store_offset_x, store_offset_y = 0.0, 0.0
            logger.warning("No bbox found in metadata; assuming local coordinates.")
        
        return sim_offset_x, sim_offset_y, store_offset_x, store_offset_y
    
    def _transform_to_sim_frame(
        self,
        positions: np.ndarray,
        offset_x: float,
        offset_y: float
    ) -> np.ndarray:
        """Transform positions to simulation frame."""
        positions_sim = positions.copy()
        if len(positions_sim) > 0 and (offset_x != 0 or offset_y != 0):
            positions_sim[:, 0] -= offset_x
            positions_sim[:, 1] -= offset_y
        return positions_sim
    
    def _process_trajectories_in_batches(
        self,
        trajectories_global: List[np.ndarray],
        scene: Any,
        site_positions_sim: np.ndarray,
        cell_ids: np.ndarray,
        sim_offset_x: float,
        sim_offset_y: float,
        store_offset_x: float,
        store_offset_y: float,
        scene_metadata: Optional[Dict] = None,
        building_map: Optional[np.ndarray] = None,
    ) -> Dict[str, List]:
        """Process UE trajectories in batches."""
        all_features = {
            'rt': [],
            'phy_fapi': [],
            'mac_rrc': [],
            'positions': [],
            'timestamps': [],
            'ue_ids': [],
            't_steps': [],
        }
        
        # Flatten all trajectory points
        all_points = []
        for ue_id, traj_global in enumerate(trajectories_global):
            for t_step, ue_pos_global in enumerate(traj_global):
                all_points.append((ue_id, t_step, ue_pos_global))
        
        batch_size = getattr(self.config, "rt_batch_size", 16)
        total_reports = 0
        dropped_reports = 0
        
        # Process in batches with adaptive splitting on RT failures
        batches = [all_points[i:i + batch_size] for i in range(0, len(all_points), batch_size)]
        pbar = tqdm(total=len(all_points), desc="Simulating UE Batches", leave=False)

        def _should_split(err: Exception) -> bool:
            def _iter_causes(ex: Exception):
                seen = set()
                while ex is not None and id(ex) not in seen:
                    seen.add(id(ex))
                    yield ex
                    ex = getattr(ex, "__cause__", None)

            for ex in _iter_causes(err):
                msg = str(ex)
                if ("Channel matrix has" in msg) or ("CFR missing" in msg) or ("Unexpected CFR shape" in msg):
                    return True
            return False

        drop_failed = getattr(self.config, "drop_failed_reports", True)
        drop_log_every = getattr(self.config, "drop_log_every", 100)
        max_resample_attempts = getattr(self.config, "max_resample_attempts", 10)
        resample_attempts = 0

        def _resample_single_point(entry: Tuple[int, int, np.ndarray]) -> Tuple[int, int, np.ndarray]:
            """Resample a single UE report inside the scene bounds."""
            if scene_metadata is None:
                raise RuntimeError("Scene metadata required for resampling failed UE positions.")
            ue_id, t_step, _ = entry
            resampled = sample_ue_trajectories(
                scene_metadata=scene_metadata,
                num_ue_per_tile=1,
                ue_height_range=self.config.ue_height_range,
                ue_velocity_range=(0.0, 0.0),
                num_reports_per_ue=1,
                report_interval_ms=self.config.report_interval_ms,
                offset=(0.0, 0.0),
                building_height_map=building_map,
                max_attempts_per_ue=getattr(self.config, "max_attempts_per_ue", 25),
                drop_failed_ue_trajectories=False,
                enforce_unique_positions=False,
                min_ue_separation_m=0.0,
                sampling_margin_m=getattr(self.config, "ue_sampling_margin_m", 0.0),
            )
            new_pos = resampled[0][0]
            return (ue_id, t_step, new_pos)

        while batches:
            batch = batches.pop(0)
            try:
                # Prepare batch inputs
                ue_positions_sim, ue_positions_store, t_steps, ue_ids = self._prepare_batch_positions(
                    batch, sim_offset_x, sim_offset_y, store_offset_x, store_offset_y
                )

                # Fix UE Z-coordinates relative to terrain
                ue_positions_sim = self._clamp_ue_to_ground(scene, ue_positions_sim, ue_positions_store)

                # Simulate batch
                rt_batch, phy_batch, mac_batch = self.batch_simulator.simulate_batch(
                    scene, ue_positions_sim, site_positions_sim, cell_ids, self.transmitter_setup
                )

                # Collect results
                all_features['rt'].append(rt_batch.to_dict())
                all_features['phy_fapi'].append(phy_batch.to_dict())
                all_features['mac_rrc'].append(mac_batch.to_dict())
                all_features['positions'].append(ue_positions_store)
                all_features['timestamps'].append(np.array(t_steps) * self.config.report_interval_ms / 1000.0)
                all_features['ue_ids'].append(np.array(ue_ids))
                all_features['t_steps'].append(np.array(t_steps))

                total_reports += len(batch)
                pbar.update(len(batch))
            except RuntimeError as e:
                if drop_failed and hasattr(e, "bad_indices") and len(batch) > 1:
                    bad_indices = sorted(set(getattr(e, "bad_indices", [])))
                    if bad_indices:
                        remaining = [entry for idx, entry in enumerate(batch) if idx not in bad_indices]
                        dropped_reports += len(bad_indices)
                        total_reports += len(bad_indices)
                        pbar.update(len(bad_indices))
                        if dropped_reports % drop_log_every == 0:
                            logger.warning(
                                "Dropped %d samples due to RT/CFR failures (most recent error: %s)",
                                dropped_reports,
                                str(e),
                            )
                        if remaining:
                            batches.insert(0, remaining)
                        continue
                if len(batch) > 1 and _should_split(e):
                    mid = len(batch) // 2
                    batches.insert(0, batch[mid:])
                    batches.insert(0, batch[:mid])
                    continue
                if len(batch) == 1 and _should_split(e):
                    if drop_failed:
                        dropped_reports += 1
                        total_reports += 1
                        pbar.update(1)
                        if dropped_reports % 50 == 0:
                            logger.warning(
                                "Dropped %d samples due to RT/CFR failures (most recent error: %s)",
                                dropped_reports,
                                str(e),
                            )
                        continue
                    resample_attempts += 1
                    if resample_attempts > max_resample_attempts:
                        pbar.close()
                        raise RuntimeError(
                            "Exceeded max_resample_attempts while retrying failed RT/CFR samples."
                        ) from e
                    logger.warning(
                        "Resampling UE report after RT/CFR failure (attempt %d/%d): %s",
                        resample_attempts,
                        max_resample_attempts,
                        str(e),
                    )
                    batches.insert(0, [_resample_single_point(batch[0])])
                    continue
                pbar.close()
                raise

        pbar.close()
        if dropped_reports > 0:
            logger.warning("Dropped %d samples total due to RT/CFR failures.", dropped_reports)
        
        return all_features
    
    def _prepare_batch_positions(
        self,
        batch: List[Tuple],
        sim_offset_x: float,
        sim_offset_y: float,
        store_offset_x: float,
        store_offset_y: float
    ) -> Tuple[np.ndarray, List, List, List]:
        """Prepare batch positions in simulation and storage frames."""
        ue_positions_sim = []
        ue_positions_store = []
        t_steps = []
        ue_ids = []
        
        for ue_id, t_step, ue_pos_global in batch:
            # Sim frame
            pos_sim = ue_pos_global.copy()
            pos_sim[0] -= sim_offset_x
            pos_sim[1] -= sim_offset_y
            ue_positions_sim.append(pos_sim)
            
            # Store frame
            pos_store = ue_pos_global.copy()
            pos_store[0] -= store_offset_x
            pos_store[1] -= store_offset_y
            ue_positions_store.append(pos_store)
            
            t_steps.append(t_step)
            ue_ids.append(ue_id)
        
        return np.array(ue_positions_sim), ue_positions_store, t_steps, ue_ids
    
    def _clamp_ue_to_ground(
        self,
        scene: Any,
        ue_positions_sim: np.ndarray,
        ue_positions_store: List
    ) -> np.ndarray:
        """Clamp UE positions to ground level."""
        if not SIONNA_AVAILABLE or scene is None:
            return ue_positions_sim
        
        try:
            import drjit as dr
            import mitsuba as mi
            mi.set_log_level(mi.LogLevel.Error)
            
            z_start = 30000.0
            batch_len = len(ue_positions_sim)
            
            # Create rays
            origin_np = np.stack([
                ue_positions_sim[:, 0],
                ue_positions_sim[:, 1],
                np.full(batch_len, z_start)
            ], axis=1).astype(np.float32)
            
            direction_np = np.tile([0.0, 0.0, -1.0], (batch_len, 1)).astype(np.float32)
            
            o = mi.Point3f(origin_np)
            d = mi.Vector3f(direction_np)
            ray = mi.Ray3f(o, d)
            
            # Ray cast
            si = scene.mi_scene.ray_intersect(ray)
            hits_t = si.t.numpy()
            valid = si.is_valid()
            ground_z = z_start - hits_t
            valid_np = valid.numpy()
            
            # Update Z
            h_ue = ue_positions_sim[:, 2]
            ue_positions_sim[valid_np, 2] = ground_z[valid_np] + h_ue[valid_np]
            
            # Fallback for invalid hits
            if not np.all(valid_np) and hasattr(scene, 'aabb'):
                z_min = float(scene.aabb[0, 2])
                ue_positions_sim[~valid_np, 2] += z_min
            
            # Sync storage Z
            for j in range(batch_len):
                ue_positions_store[j][2] = float(ue_positions_sim[j, 2])
                
        except Exception as e:
            logger.debug(f"UE Ground Clamping failed: {e}")
            # Fallback to AABB
            try:
                if hasattr(scene, 'aabb'):
                    z_min = float(scene.aabb[0, 2])
                    ue_positions_sim[:, 2] += z_min
                    for j in range(len(ue_positions_store)):
                        ue_positions_store[j][2] = ue_positions_sim[j, 2]
            except:
                pass
        
        return ue_positions_sim
    
    # Keep map generation methods (delegated to existing generators)
    def _generate_radio_map_for_scene(self, scene: Any, scene_path: Path, scene_metadata: Dict):
        """Generate radio map - delegates to RadioMapGenerator."""
        if self.radio_map_generator is None:
            return
            
        try:
            logger.info(f"Generating Radio Map for scene {scene_metadata.get('scene_id', 'unknown')}...")
            
            # Extract cell sites from metadata
            sites = scene_metadata.get('sites', [])
            
            # Generate and internally save if configured
            # RadioMapGenerator.generate_for_scene returns the Sionna RadioMap object if return_sionna_object=True.
            # However, looking at the call signature in our earlier listing, it seems we need to explicitly ask for it.
            # But wait, does generate_for_scene return the object by default? 
            # The signature was: generate_for_scene(..., return_sionna_object=False) -> Optional[Any]
            
            # Let's request the object so we can use it for visualization overlay
            radio_map_obj = self.radio_map_generator.generate_for_scene(
                scene=scene,
                cell_sites=sites,
                show_progress=True,
                return_sionna_object=True
            )
            
            # Visualization
            scene_id = scene_metadata.get('scene_id', 'unknown')
            viz_dir = self.config.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 3D Renders (Top-Down, ISO) with optional coverage overlay
            logger.info(f"Rendering 3D visualizations for {scene_id}...")
            render_scene_3d(
                scene=scene,
                scene_id=scene_id,
                metadata=scene_metadata,
                output_dir=viz_dir / "3d",
                radio_map=radio_map_obj
            )
            
            # 2. 2D Map Plots (if we have the raw map data array easily)
            # The radio_map_obj from Sionna might be complex. 
            # Ideally we'd pass the numpy array to save_map_visualizations.
            # If radio_map_obj is a Sionna object, we can extract it.
            # Or we can rely on render_scene_3d doing the heavy lifting for radio maps.
            
            # Let's try to extract the numpy array from the radio map object for save_map_visualizations
            # if available, otherwise just skip the 2D plot of radio content here (as render_scene_3d covers it).
            # But save_map_visualizations handles both radio and OSM.
            
            if radio_map_obj is not None:
                # Extract radio map data for 2D visualization
                # Sionna CoverageMap stores data in .as_tensor() or we can access it if converted
                # Our RadioMapGenerator wraps Sionna's CoverageMap.
                # If return_sionna_object=True, we get the Sionna object.
                
                # Check for common attributes to get the data as numpy
                radio_data = None
                try:
                    if hasattr(radio_map_obj, 'as_tensor'):
                        # typical Sionna CoverageMap
                        tensor = radio_map_obj.as_tensor()
                        if hasattr(tensor, 'cpu'):
                            radio_data = tensor.cpu().numpy()
                        else:
                            radio_data = tensor.numpy()
                    elif hasattr(radio_map_obj, 'rssi'):
                         # Maybe raw RSSI tensor
                         tensor = radio_map_obj.rssi
                         if hasattr(tensor, 'cpu'):
                            radio_data = tensor.cpu().numpy()
                         else:
                            radio_data = tensor.numpy()
                    
                    if radio_data is not None:
                        # radio_data shape typically [num_tx, num_rx_h, num_rx_w]
                        # save_map_visualizations expects [C, H, W]
                        logger.info(f"Saving 2D Radio Map plots for {scene_id}...")
                        save_map_visualizations(
                            scene_id=scene_id,
                            radio_map=radio_data,
                            osm_map=None, # Already handled in OSM generation or handled separately
                            output_dir=viz_dir
                        )
                except Exception as viz_e:
                    logger.warning(f"Could not extract radio data for 2D plot: {viz_e}")
            
        except Exception as e:
            logger.error(f"Failed to generate radio map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _generate_osm_map_for_scene(self, scene: Any, metadata: Dict):
        """Generate OSM map - delegates to OSMRasterizer."""
        if self.osm_rasterizer is None:
            return
            
        try:
            scene_id = metadata.get('scene_id', 'unknown')
            logger.info(f"Generating OSM Map for scene {scene_id}...")
            
            # Rasterize with metadata for road extraction
            osm_map = self.osm_rasterizer.rasterize(scene, scene_metadata=metadata)
            
            # Save to disk
            # Ensure osm_maps directory exists
            osm_dir = self.config.output_dir / "osm_maps"
            osm_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = osm_dir / f"{scene_id}.npy"
            np.save(output_path, osm_map)
            logger.info(f"Saved OSM map to {output_path}")
            
            # Visualization
            viz_dir = self.config.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving 2D map visualizations for {scene_id}...")
            save_map_visualizations(
                scene_id=scene_id,
                radio_map=None, # Passed separately in radio generation block or extracted
                osm_map=osm_map,
                output_dir=viz_dir
            )
                
        except Exception as e:
            logger.error(f"Failed to generate OSM map: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _simulate_mock(self, ue_positions, site_positions, cell_ids):
        """Compatibility wrapper for tests."""
        return self.batch_simulator._simulate_mock_batch(ue_positions, site_positions, cell_ids)
