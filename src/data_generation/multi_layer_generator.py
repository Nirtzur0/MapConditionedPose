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

# Try importing Zarr writer (optional)
try:
    from .zarr_writer import ZarrDatasetWriter
    from .osm_rasterizer import OSMRasterizer
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    logger.warning("Zarr not available; dataset writing will fail without the zarr package.")

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

        self.native_extractor = SionnaNativeKPIExtractor(
            carrier_frequency_hz=config.carrier_frequency_hz,
            bandwidth_hz=config.bandwidth_hz,
            noise_figure_db=config.noise_figure_db
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
        self.data_stacker = DataStacker(max_neighbors=config.max_neighbors)
        self.measurement_processor = MeasurementProcessor(config=config)
        
        # Initialize Zarr writer
        if ZARR_AVAILABLE:
            self.zarr_writer = ZarrDatasetWriter(
                output_dir=config.output_dir,
                chunk_size=config.zarr_chunk_size,
            )
        else:
            logger.warning("Zarr writer unavailable; install 'zarr' for dataset writing.")
        
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
    
    def generate_dataset(self,
                        scene_ids: Optional[List[str]] = None,
                        num_scenes: Optional[int] = None) -> Path:
        """
        Generates a complete dataset from M1 scenes.
        
        Args:
            scene_ids: Specific scene IDs to process (defaults to all).
            num_scenes: Limits the number of scenes processed (for testing).
            
        Returns:
            Path to the generated Zarr dataset.
        """
        # Discover scenes
        if scene_ids is None:
            scene_dirs = [d for d in self.config.scene_dir.iterdir() if d.is_dir()]
            scene_ids = [d.name for d in scene_dirs if (d / 'scene.xml').exists()]
        
        if num_scenes:
            scene_ids = scene_ids[:num_scenes]
        
        logger.info(f"Processing {len(scene_ids)} scenes...")
        
        # Process each scene
        for scene_id in tqdm(scene_ids, desc="Generating Dataset"):
            scene_path = self.config.scene_dir / scene_id / 'scene.xml'
            
            if not scene_path.exists():
                logger.warning(f"Scene file not found: {scene_path}")
                continue
            
            try:
                scene_data = self.generate_scene_data(scene_path, scene_id)
                
                # Write to Zarr
                if ZARR_AVAILABLE and scene_data:
                    self.zarr_writer.append(scene_data, scene_id=scene_id)
                    
            except Exception as e:
                logger.error(f"Failed to process scene {scene_id}: {e}")
                logger.error(traceback.format_exc())
        
        # Finalize dataset
        if ZARR_AVAILABLE:
            output_path = self.zarr_writer.finalize()
            logger.info(f"Dataset saved to: {output_path}")
            return output_path
        
        return None
    
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
        trajectories_global = sample_ue_trajectories(
            scene_metadata=scene_metadata,
            num_ue_per_tile=self.config.num_ue_per_tile,
            ue_height_range=self.config.ue_height_range,
            ue_velocity_range=self.config.ue_velocity_range,
            num_reports_per_ue=self.config.num_reports_per_ue,
            report_interval_ms=self.config.report_interval_ms,
            offset=(0.0, 0.0)
        )
        
        # Process trajectories in batches
        all_features = self._process_trajectories_in_batches(
            trajectories_global,
            scene,
            site_positions_sim,
            cell_ids,
            sim_offset_x,
            sim_offset_y,
            store_offset_x,
            store_offset_y
        )
        
        # Stack into arrays
        scene_data = self.data_stacker.stack_scene_data(all_features)
        
        # Apply measurement realism
        scene_data = self.measurement_processor.apply_realism(scene_data)

        # Log statistics
        stats = self.batch_simulator.get_statistics()
        total = stats['sionna_ok'] + stats['sionna_fail']
        if total > 0:
            logger.info(f"Sionna RT summary: {stats['sionna_ok']}/{total} successful, {stats['sionna_fail']} mock fallbacks")
        
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
        store_offset_y: float
    ) -> Dict[str, List]:
        """Process UE trajectories in batches."""
        all_features = {
            'rt': [], 'phy_fapi': [], 'mac_rrc': [], 'positions': [], 'timestamps': []
        }
        
        # Flatten all trajectory points
        all_points = []
        for traj_global in trajectories_global:
            for t_step, ue_pos_global in enumerate(traj_global):
                all_points.append((t_step, ue_pos_global))
        
        batch_size = 32
        total_reports = 0
        
        # Process in batches
        for i in tqdm(range(0, len(all_points), batch_size), desc="Simulating UE Batches", leave=False):
            batch = all_points[i:i + batch_size]
            
            # Prepare batch inputs
            ue_positions_sim, ue_positions_store, t_steps = self._prepare_batch_positions(
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
            
            total_reports += len(batch)
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {total_reports}/{len(all_points)} measurements...")
        
        return all_features
    
    def _prepare_batch_positions(
        self,
        batch: List[Tuple],
        sim_offset_x: float,
        sim_offset_y: float,
        store_offset_x: float,
        store_offset_y: float
    ) -> Tuple[np.ndarray, List, List]:
        """Prepare batch positions in simulation and storage frames."""
        ue_positions_sim = []
        ue_positions_store = []
        t_steps = []
        
        for t_step, ue_pos_global in batch:
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
        
        return np.array(ue_positions_sim), ue_positions_store, t_steps
    
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
        # This method remains unchanged as it already uses RadioMapGenerator
        pass
    
    def _generate_osm_map_for_scene(self, scene: Any, metadata: Dict):
        """Generate OSM map - delegates to OSMRasterizer."""
        # This method remains unchanged as it already uses OSMRasterizer
        pass
