"""
Multi-Layer Data Generator
Orchestrates RT simulation, feature extraction, and dataset generation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import yaml
import json

from .features import (
    RTFeatureExtractor, PHYFAPIFeatureExtractor, MACRRCFeatureExtractor,
    RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures
)
from .measurement_utils import add_measurement_dropout

logger = logging.getLogger(__name__)

# Try importing Zarr writer (optional)
try:
    from .zarr_writer import ZarrDatasetWriter
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available - dataset writing will fail without zarr package")

# Try importing Sionna
try:
    import sionna
    from sionna.rt import Scene, Camera, load_scene
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available - MultiLayerDataGenerator will operate in mock mode")


@dataclass
class DataGenerationConfig:
    """Configuration for multi-layer data generation."""
    
    # Scene and tile info
    scene_dir: Path
    scene_metadata_path: Path
    
    # RF parameters
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    tx_power_dbm: float = 43.0
    noise_figure_db: float = 9.0
    
    # Sionna RT parameters
    use_mock_mode: bool = False  # Toggle between real and mock
    max_depth: int = 5
    num_samples: int = 1_000_000
    enable_diffraction: bool = True
    
    # Sampling strategy
    num_ue_per_tile: int = 100
    ue_height_range: Tuple[float, float] = (1.5, 1.5)  # Fixed pedestrian height
    ue_velocity_range: Tuple[float, float] = (0.0, 1.5)  # 0-1.5 m/s
    
    # Temporal sequence
    num_reports_per_ue: int = 10
    report_interval_ms: float = 200.0  # 200 ms between measurements
    
    # Feature extraction
    enable_k_factor: bool = False
    enable_beam_management: bool = True
    num_beams: int = 64
    max_neighbors: int = 8
    
    # Measurement realism
    measurement_dropout_rates: Dict[str, float] = None
    quantization_enabled: bool = True
    
    # Output
    output_dir: Path = Path("data/synthetic")
    zarr_chunk_size: int = 100
    
    def __post_init__(self):
        if self.measurement_dropout_rates is None:
            # Default 3GPP-like dropout rates
            self.measurement_dropout_rates = {
                'rsrp': 0.05,  # Serving cell: low dropout
                'rsrq': 0.10,
                'sinr': 0.10,
                'cqi': 0.15,
                'ri': 0.20,  # Less frequent reporting
                'pmi': 0.25,
                'neighbor_rsrp': 0.30,  # Neighbors: higher dropout
            }
        
        # Convert paths to Path objects
        self.scene_dir = Path(self.scene_dir)
        self.scene_metadata_path = Path(self.scene_metadata_path)
        self.output_dir = Path(self.output_dir)
        
    @classmethod
    def from_yaml(cls, path: str) -> 'DataGenerationConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class MultiLayerDataGenerator:
    """
    Orchestrates end-to-end data generation pipeline:
    1. Load M1 scene in Sionna RT
    2. Sample UE positions and trajectories
    3. Run ray tracing (RT layer)
    4. Extract PHY/FAPI features (L2)
    5. Extract MAC/RRC features (L3)
    6. Apply measurement realism (dropout, quantization)
    7. Save to Zarr dataset
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Args:
            config: Data generation configuration
        """
        self.config = config
        
        # Initialize feature extractors
        self.rt_extractor = RTFeatureExtractor(
            carrier_frequency_hz=config.carrier_frequency_hz,
            bandwidth_hz=config.bandwidth_hz,
            compute_k_factor=config.enable_k_factor,
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
        )
        
        # Initialize Zarr writer (only if available)
        self.zarr_writer = None
        if ZARR_AVAILABLE:
            self.zarr_writer = ZarrDatasetWriter(
                output_dir=config.output_dir,
                chunk_size=config.zarr_chunk_size,
            )
        else:
            logger.warning("Zarr writer not available - install zarr package for dataset writing")
        
        # Load scene metadata
        self.scene_metadata = self._load_scene_metadata()
        self._rx_counter = 0
        self._logged_sionna = False
        self._sionna_ok = 0
        self._sionna_fail = 0
        
        logger.info(f"MultiLayerDataGenerator initialized: {config.scene_dir}")
        logger.info(f"  Carrier freq: {config.carrier_frequency_hz/1e9:.2f} GHz")
        logger.info(f"  UEs per tile: {config.num_ue_per_tile}")
        logger.info(f"  Reports per UE: {config.num_reports_per_ue}")
        logger.info(f"  Sionna RT: {'enabled' if SIONNA_AVAILABLE else 'disabled'}")
    
    def _load_scene_metadata(self) -> Dict:
        """Load scene metadata from M1 output."""
        metadata_path = self.config.scene_metadata_path
        if not metadata_path.exists():
            logger.warning(f"Scene metadata not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded scene metadata: {len(metadata.get('sites', []))} sites")
        return metadata
    
    def generate_dataset(self, 
                        scene_ids: Optional[List[str]] = None,
                        num_scenes: Optional[int] = None) -> Path:
        """
        Generate complete dataset from M1 scenes.
        
        Args:
            scene_ids: List of scene IDs to process (if None, process all)
            num_scenes: Limit number of scenes (for testing)
            
        Returns:
            output_path: Path to Zarr dataset
        """
        # Find all scenes
        if scene_ids is None:
            scene_dirs = sorted(self.config.scene_dir.glob("scene_*"))
            scene_ids = [d.name for d in scene_dirs]
        
        if num_scenes is not None:
            scene_ids = scene_ids[:num_scenes]
        
        logger.info(f"Generating dataset from {len(scene_ids)} scenes...")
        
        # Process each scene
        for i, scene_id in enumerate(scene_ids):
            logger.info(f"Processing scene {i+1}/{len(scene_ids)}: {scene_id}")
            
            scene_path = self.config.scene_dir / scene_id / "scene.xml"
            if not scene_path.exists():
                logger.warning(f"Scene not found: {scene_path}, skipping")
                continue
            
            # Generate data for this scene
            scene_data = self.generate_scene_data(scene_path, scene_id)
            
            # Write to Zarr
            if self.zarr_writer is not None:
                self.zarr_writer.append(scene_data, scene_id=scene_id)
            else:
                logger.warning("Zarr writer not available, skipping data write")
        
        # Finalize dataset
        if self.zarr_writer is not None:
            output_path = self.zarr_writer.finalize()
            logger.info(f"Dataset generation complete: {output_path}")
            return output_path
        else:
            logger.warning("Dataset generation complete but no data written (zarr not available)")
            return self.config.output_dir
    
    def generate_scene_data(self, 
                           scene_path: Path,
                           scene_id: str) -> Dict[str, np.ndarray]:
        """
        Generate multi-layer data for a single scene.
        
        Args:
            scene_path: Path to scene.xml
            scene_id: Scene identifier
            
        Returns:
            scene_data: Dict with all features and positions
        """
        # Load scene in Sionna
        if SIONNA_AVAILABLE:
            scene = self._load_sionna_scene(scene_path)
        else:
            logger.warning("Sionna not available - using mock data")
            scene = None
        
        # Load site positions and cell IDs from metadata
        scene_metadata = self.scene_metadata.get(scene_id, {})
        sites = scene_metadata.get('sites', [])
        site_positions = np.array([s['position'] for s in sites])
        cell_ids = np.array([s['cell_id'] for s in sites])
        
        if len(sites) == 0:
            logger.warning(f"No sites in metadata for {scene_id}, using mock")
            site_positions = np.array([[0, 0, 30], [500, 0, 30]])
            cell_ids = np.array([1, 2])
        
        # Setup transmitters in Sionna
        if scene is not None:
            self._setup_transmitters(scene, site_positions, sites if sites else {})
        
        # Sample UE positions and trajectories
        ue_trajectories = self._sample_ue_trajectories(scene_metadata)
        
        # Collect data for all UEs and time steps
        all_data = {
            'rt': [],
            'phy_fapi': [],
            'mac_rrc': [],
            'positions': [],
            'timestamps': [],
        }
        
        num_ues = len(ue_trajectories)
        for ue_idx, trajectory in enumerate(ue_trajectories):
            if (ue_idx + 1) % 10 == 0:
                logger.info(f"  Processing UE {ue_idx+1}/{num_ues}")
            
            # Generate temporal sequence
            for t_idx, ue_pos in enumerate(trajectory):
                # Run simulation for this UE position
                rt_features, phy_features, mac_features = self._simulate_measurement(
                    scene, ue_pos, site_positions, cell_ids
                )
                
                # Store features
                all_data['rt'].append(rt_features.to_dict())
                all_data['phy_fapi'].append(phy_features.to_dict())
                all_data['mac_rrc'].append(mac_features.to_dict())
                all_data['positions'].append(ue_pos)
                
                # Timestamp
                timestamp = t_idx * self.config.report_interval_ms / 1000.0  # seconds
                all_data['timestamps'].append(timestamp)
        
        # Stack into arrays
        scene_data = self._stack_scene_data(all_data)
        
        # Apply measurement realism
        scene_data = self._apply_measurement_realism(scene_data)

        total = self._sionna_ok + self._sionna_fail
        if total > 0:
            logger.info(f"Sionna RT summary: {self._sionna_ok}/{total} successful, {self._sionna_fail} mock fallbacks")
        
        return scene_data
    
    def _sample_ue_trajectories(self, scene_metadata: Dict) -> List[np.ndarray]:
        """
        Sample UE positions and trajectories.
        
        Args:
            scene_metadata: Scene metadata with bounding box
            
        Returns:
            trajectories: List of [num_reports, 3] position arrays
        """
        # Get scene bounds
        bbox = scene_metadata.get('bbox', {})
        x_min = bbox.get('x_min', -500)
        x_max = bbox.get('x_max', 500)
        y_min = bbox.get('y_min', -500)
        y_max = bbox.get('y_max', 500)
        
        trajectories = []
        for _ in range(self.config.num_ue_per_tile):
            # Sample initial position
            x0 = np.random.uniform(x_min, x_max)
            y0 = np.random.uniform(y_min, y_max)
            z0 = np.random.uniform(*self.config.ue_height_range)
            
            # Sample velocity
            speed = np.random.uniform(*self.config.ue_velocity_range)
            direction = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(direction)
            vy = speed * np.sin(direction)
            
            # Generate trajectory
            trajectory = []
            for t in range(self.config.num_reports_per_ue):
                dt = t * self.config.report_interval_ms / 1000.0
                x = x0 + vx * dt
                y = y0 + vy * dt
                z = z0  # Fixed height
                
                # Clip to bounds
                x = np.clip(x, x_min, x_max)
                y = np.clip(y, y_min, y_max)
                
                trajectory.append([x, y, z])
            
            trajectories.append(np.array(trajectory))
        
        return trajectories
    
    def _simulate_measurement(self,
                             scene: Any,
                             ue_position: np.ndarray,
                             site_positions: np.ndarray,
                             cell_ids: np.ndarray) -> Tuple[RTLayerFeatures, 
                                                             PHYFAPILayerFeatures,
                                                             MACRRCLayerFeatures]:
        """
        Run simulation for single UE position.
        
        Args:
            scene: Sionna RT Scene
            ue_position: [3] UE position (x, y, z)
            site_positions: [num_sites, 3] site positions
            cell_ids: [num_sites] cell IDs
            
        Returns:
            (rt_features, phy_features, mac_features)
        """
        if not SIONNA_AVAILABLE or scene is None:
            logger.debug("Sionna not available or no scene, using mock simulation")
            return self._simulate_mock(ue_position, site_positions, cell_ids)
        
        rx_name = None
        try:
            # Setup receiver at UE position
            rx_name = f"UE_{self._rx_counter}"
            self._rx_counter += 1
            rx_name = self._setup_receiver(scene, ue_position, rx_name)
            
            # Compute propagation paths using Sionna RT
            logger.debug(f"Computing paths for UE at {ue_position}")
            from sionna.rt import PathSolver
            from mitsuba import Float
            
            max_depth = getattr(self.config, 'max_depth', 5)
            samples_per_src = getattr(self.config, 'num_samples', 100_000)
            max_num_paths = getattr(self.config, 'max_num_paths', 100_000)
            paths = PathSolver()(
                scene,
                max_depth=max_depth,
                samples_per_src=samples_per_src,
                max_num_paths_per_src=max_num_paths,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=True,
                diffraction=getattr(self.config, 'enable_diffraction', True),
                edge_diffraction=False
            )
            if not self._logged_sionna:
                try:
                    valid_paths = int(paths.valid.numpy().sum())
                except Exception:
                    valid_paths = 0
                logger.info(f"Sionna RT paths computed: {valid_paths}")
                self._logged_sionna = True
            
            # Extract RT features
            rt_features = self.rt_extractor.extract(paths)
            
            # Compute channel matrices (frequency response across subcarriers)
            channel_matrix = None
            try:
                num_subcarriers = getattr(self.config, 'num_subcarriers', 100)
                freqs = np.linspace(
                    self.config.carrier_frequency_hz - self.config.bandwidth_hz / 2,
                    self.config.carrier_frequency_hz + self.config.bandwidth_hz / 2,
                    num_subcarriers
                )
                cfr = paths.cfr(
                    frequencies=Float(list(freqs)),
                    out_type='numpy'
                )
                channel_matrix = cfr[np.newaxis, ...]
            except Exception:
                channel_matrix = None
            
            if rt_features.is_mock:
                self._sionna_fail += 1
            else:
                self._sionna_ok += 1

            # Extract PHY features
            phy_features = self.phy_extractor.extract(
                rt_features=rt_features,
                channel_matrix=channel_matrix,
                interference_matrices=None
            )
            
            # Extract SYS features
            mac_features = self.mac_extractor.extract(
                phy_features=phy_features,
                ue_positions=ue_position[np.newaxis, :],
                site_positions=site_positions,
                cell_ids=cell_ids
            )
            
            return rt_features, phy_features, mac_features
            
        except Exception as e:
            self._sionna_fail += 1
            if self._sionna_fail <= 3:
                logger.error(f"Sionna simulation failed: {e}")
                if self._sionna_fail == 3:
                    logger.warning("Suppressing further Sionna errors (see log file for full details)")
            logger.warning("Falling back to mock simulation")
            return self._simulate_mock(ue_position, site_positions, cell_ids)
        finally:
            if rx_name:
                try:
                    scene.remove(rx_name)
                except Exception:
                    pass
    
    def _load_sionna_scene(self, scene_path: Path) -> Any:
        """
        Load Mitsuba XML scene into Sionna RT.
        
        Args:
            scene_path: Path to scene.xml (Mitsuba format)
            
        Returns:
            Sionna RT Scene object
        """
        if not SIONNA_AVAILABLE:
            logger.warning("Sionna not available, cannot load scene")
            return None
            
        from sionna.rt import load_scene
        
        logger.info(f"Loading Sionna scene from {scene_path}")
        scene = load_scene(str(scene_path))
        
        # Apply scene-level settings
        scene.frequency = self.config.carrier_frequency_hz
        scene.synthetic_array = True
        
        logger.info(f"Scene loaded: frequency={self.config.carrier_frequency_hz/1e9:.2f} GHz")
        return scene
    
    def _setup_transmitters(self, scene: Any, 
                           site_positions: np.ndarray,
                           site_metadata: Dict) -> List[Any]:
        """
        Setup cell site transmitters with antenna arrays.
        
        Args:
            scene: Sionna RT Scene
            site_positions: [num_sites, 3] positions (x, y, z) in meters
            site_metadata: Site configuration
            
        Returns:
            List of Sionna Transmitter objects
        """
        if not SIONNA_AVAILABLE or scene is None:
            return []
            
        from sionna.rt import Transmitter, PlanarArray
        
        transmitters = []
        
        # Create antenna array based on frequency (shared across transmitters)
        if self.config.carrier_frequency_hz < 10e9:
            # Sub-6 GHz: 8x8 array, vertical polarization
            array = PlanarArray(
                num_rows=8, num_cols=8,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V"
            )
        else:
            # mmWave: 16x16 array, vertical polarization
            array = PlanarArray(
                num_rows=16, num_cols=16,
                vertical_spacing=0.5, horizontal_spacing=0.5,
                pattern="iso", polarization="V"
            )
        scene.tx_array = array

        for site_idx, pos in enumerate(site_positions):
            
            
            # Get site-specific metadata
            if isinstance(site_metadata, list) and site_idx < len(site_metadata):
                meta = site_metadata[site_idx]
                azimuth = meta.get('orientation', [0, 0, 0])[0] if 'orientation' in meta else 0.0
                downtilt = meta.get('orientation', [0, 10, 0])[1] if 'orientation' in meta else 10.0
            else:
                azimuth = site_metadata.get(f'site_{site_idx}_azimuth', 0.0)
                downtilt = site_metadata.get(f'site_{site_idx}_downtilt', 10.0)
            
            tx = Transmitter(
                name=f"BS_{site_idx}",
                position=pos if isinstance(pos, list) else pos.tolist(),
                orientation=[azimuth, downtilt, 0.0]
            )
            
            scene.add(tx)
            transmitters.append(tx)
            logger.debug(f"Added transmitter BS_{site_idx} at {pos}")
        
        return transmitters
    
    def _setup_receiver(self, scene: Any, ue_position: np.ndarray, name: str) -> str:
        """
        Setup UE receiver at given position.
        
        Args:
            scene: Sionna RT Scene
            ue_position: [3] position in meters
            
        Returns:
            Receiver name
        """
        if not SIONNA_AVAILABLE or scene is None:
            return None
            
        from sionna.rt import Receiver, PlanarArray
        
        ue_array = PlanarArray(
            num_rows=1, num_cols=2,
            vertical_spacing=0.5, horizontal_spacing=0.5,
            pattern="iso", polarization="V"
        )
        scene.rx_array = ue_array
        
        rx = Receiver(
            name=name,
            position=ue_position.tolist(),
            orientation=[0.0, 0.0, 0.0]
        )
        
        scene.add(rx)
        return name
    
    def _simulate_mock(self,
                      ue_position: np.ndarray,
                      site_positions: np.ndarray,
                      cell_ids: np.ndarray) -> Tuple[RTLayerFeatures,
                                                      PHYFAPILayerFeatures,
                                                      MACRRCLayerFeatures]:
        """Mock simulation for testing."""
        # RT features
        rt_features = self.rt_extractor._extract_mock()
        
        # Reshape to single UE
        for key in ['path_gains', 'path_delays', 'path_aoa_azimuth', 
                   'path_aoa_elevation', 'path_aod_azimuth', 'path_aod_elevation',
                   'path_doppler']:
            setattr(rt_features, key, getattr(rt_features, key)[0:1, 0:1])
        rt_features.rms_delay_spread = rt_features.rms_delay_spread[0:1, 0:1]
        if rt_features.k_factor is not None:
            rt_features.k_factor = rt_features.k_factor[0:1, 0:1]
        rt_features.num_paths = rt_features.num_paths[0:1, 0:1]
        
        # PHY features
        phy_features = self.phy_extractor.extract(rt_features)
        
        # Expand to multi-cell
        num_cells = len(cell_ids)
        phy_features.rsrp = np.random.uniform(-100, -60, (1, 1, num_cells))
        phy_features.rsrq = np.random.uniform(-15, -5, (1, 1, num_cells))
        phy_features.sinr = np.random.uniform(-5, 25, (1, 1, num_cells))
        phy_features.cqi = np.random.randint(0, 16, (1, 1, num_cells))
        phy_features.ri = np.random.randint(1, 5, (1, 1, num_cells))
        phy_features.pmi = np.random.randint(0, 8, (1, 1, num_cells))
        
        # MAC/RRC features
        ue_positions_batch = ue_position[np.newaxis, np.newaxis, :]  # [1, 1, 3]
        mac_features = self.mac_extractor.extract(
            phy_features, ue_positions_batch, site_positions, cell_ids
        )
        
        return rt_features, phy_features, mac_features
    
    def _stack_scene_data(self, all_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Stack lists of features into arrays."""
        stacked = {}
        
        # Positions and timestamps (simple stack)
        stacked['positions'] = np.array(all_data['positions'])  # [N, 3]
        stacked['timestamps'] = np.array(all_data['timestamps'])  # [N]
        
        # RT features
        for key in all_data['rt'][0].keys():
            arrays = [d[key] for d in all_data['rt']]
            # Handle variable shapes (squeeze batch/rx dims)
            arrays_squeezed = [np.squeeze(a) if a.size > 0 else a for a in arrays]
            
            try:
                stacked[key] = np.stack(arrays_squeezed, axis=0)
            except ValueError:
                # Variable length (e.g., paths) - keep as list
                stacked[key] = arrays_squeezed
                logger.warning(f"Variable length array for {key}, storing as list")
        
        # PHY/FAPI features
        for key in all_data['phy_fapi'][0].keys():
            arrays = [d[key] for d in all_data['phy_fapi']]
            arrays_squeezed = [np.squeeze(a) if a is not None and a.size > 0 else a 
                              for a in arrays]
            
            if arrays_squeezed[0] is not None:
                try:
                    stacked[key] = np.stack(arrays_squeezed, axis=0)
                except (ValueError, TypeError):
                    stacked[key] = arrays_squeezed
        
        # MAC/RRC features
        for key in all_data['mac_rrc'][0].keys():
            arrays = [d[key] for d in all_data['mac_rrc']]
            arrays_squeezed = [np.squeeze(a) if a is not None and a.size > 0 else a 
                              for a in arrays]
            
            if arrays_squeezed[0] is not None:
                try:
                    stacked[key] = np.stack(arrays_squeezed, axis=0)
                except (ValueError, TypeError):
                    stacked[key] = arrays_squeezed
        
        return stacked
    
    def _apply_measurement_realism(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply measurement dropout and quantization.
        
        Args:
            scene_data: Scene data dictionary
            
        Returns:
            scene_data_realistic: Data with dropout and quantization
        """
        if not self.config.quantization_enabled:
            return scene_data
        
        # Apply dropout to PHY/FAPI measurements
        phy_fapi_keys = [k for k in scene_data.keys() if k.startswith('phy_fapi/')]
        phy_fapi_dict = {k: scene_data[k] for k in phy_fapi_keys if isinstance(scene_data[k], np.ndarray)}
        
        # Map keys to dropout rates
        dropout_mapping = {
            'phy_fapi/rsrp': 'rsrp',
            'phy_fapi/rsrq': 'rsrq',
            'phy_fapi/sinr': 'sinr',
            'phy_fapi/cqi': 'cqi',
            'phy_fapi/ri': 'ri',
            'phy_fapi/pmi': 'pmi',
        }
        
        dropout_rates = {k: self.config.measurement_dropout_rates.get(v, 0.0)
                        for k, v in dropout_mapping.items()}
        
        phy_fapi_dropped = add_measurement_dropout(phy_fapi_dict, dropout_rates, seed=42)
        
        # Update scene_data
        scene_data.update(phy_fapi_dropped)
        
        logger.debug(f"Applied measurement dropout: "
                    f"{sum(dropout_rates.values())/len(dropout_rates)*100:.1f}% avg")
        
        return scene_data


if __name__ == "__main__":
    # Test data generator
    logger.info("Testing MultiLayerDataGenerator...")
    
    # Create test config
    config = DataGenerationConfig(
        scene_dir=Path("test_scenes"),
        scene_metadata_path=Path("test_scenes/metadata.json"),
        num_ue_per_tile=10,
        num_reports_per_ue=5,
        output_dir=Path("test_output"),
    )
    
    # Create mock scene directory
    config.scene_dir.mkdir(exist_ok=True)
    config.output_dir.mkdir(exist_ok=True)
    
    # Write mock metadata
    metadata = {
        'scene_001': {
            'bbox': {'x_min': -500, 'x_max': 500, 'y_min': -500, 'y_max': 500},
            'sites': [
                {'position': [0, 0, 30], 'cell_id': 1},
                {'position': [500, 0, 30], 'cell_id': 2},
            ],
        }
    }
    with open(config.scene_metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Initialize generator
    generator = MultiLayerDataGenerator(config)
    
    # Test UE trajectory sampling
    trajectories = generator._sample_ue_trajectories(metadata['scene_001'])
    logger.info(f"✓ Sampled {len(trajectories)} UE trajectories")
    logger.info(f"  Trajectory shape: {trajectories[0].shape}")
    
    # Test single measurement simulation
    ue_pos = np.array([100, 200, 1.5])
    site_positions = np.array([[0, 0, 30], [500, 0, 30]])
    cell_ids = np.array([1, 2])
    rt_feat, phy_feat, mac_feat = generator._simulate_mock(ue_pos, site_positions, cell_ids)
    logger.info(f"✓ Simulated single measurement")
    logger.info(f"  RT features: {len(rt_feat.to_dict())} arrays")
    logger.info(f"  PHY features: {len(phy_feat.to_dict())} arrays")
    logger.info(f"  MAC features: {len(mac_feat.to_dict())} arrays")
    
    logger.info("\nMultiLayerDataGenerator tests passed! ✓")
