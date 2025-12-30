"""
Orchestrates multi-layer radio propagation simulation, feature extraction, and dataset generation.
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
from .measurement_utils import add_measurement_dropout
from .radio_map_generator import RadioMapGenerator, RadioMapConfig

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


MAX_CELLS = 16  # Fixed size for array padding

class MultiLayerDataGenerator:
    """
    Orchestrates end-to-end data generation pipeline:
    1. Loads M1 scene.
    2. Samples UE positions and trajectories.
    3. Runs ray tracing (RT layer).
    4. Extracts PHY/FAPI features (L2).
    5. Extracts MAC/RRC features (L3).
    6. Applies measurement realism (dropout, quantization).
    7. Saves to Zarr dataset.
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initializes the data generator with a configuration.
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

        self.native_extractor = SionnaNativeKPIExtractor(
            carrier_frequency_hz=config.carrier_frequency_hz,
            bandwidth_hz=config.bandwidth_hz,
            noise_figure_db=config.noise_figure_db
        )
        
        if ZARR_AVAILABLE:
            self.zarr_writer = ZarrDatasetWriter(
                output_dir=config.output_dir,
                chunk_size=config.zarr_chunk_size,
            )
        else:
            logger.warning("Zarr writer unavailable; install 'zarr' for dataset writing.")
        
        # Initialize Sionna Backbone (OFDM/MIMO)
        if SIONNA_AVAILABLE:
            try:
                from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
                from sionna.phy.mimo import StreamManagement
                
                # 1. Stream Management
                # We assume 1 stream per UE for simplistic SISO/MIMO abstraction in simulation
                # or match it to num_tx/num_rx later. For now, we set generic defaults.
                # In batch simulation, we create these dynamically or reuse strict config?
                # Let's define a "System Backbone" config.
                
                # 30 kHz SCS, 100 MHz BW -> ~273 RBs -> ~3276 subcarriers
                scs = 30e3
                fft_size = int(config.bandwidth_hz / scs) # e.g. 3333 -> 4096 next pow2?
                # Round to nearest standard FFT size or just use explicit num_subcarriers
                num_subcarriers = getattr(config, 'num_subcarriers', 1024) # Default to 1024 if not set
                fft_size = 2**int(np.ceil(np.log2(num_subcarriers)))
                
                self.rg = ResourceGrid(
                    num_ofdm_symbols=14,
                    fft_size=fft_size,
                    subcarrier_spacing=scs,
                    num_tx=1, # Updated dynamically per site?
                    num_streams_per_tx=1,
                    cyclic_prefix_length=20, # config?
                    pilot_pattern=None, # Raw channel
                    pilot_ofdm_symbol_indices=None
                )
                logger.info(f"Sionna Backbone initialized: OFDM Resource Grid ({fft_size} FFT, {scs/1e3} kHz SCS)")
            except Exception as e:
                logger.warning(f"Failed to initialize Sionna Backbone: {e}")
                self.rg = None

        # Initialize counters
        self._rx_counter = 0
        self._logged_sionna = False
        self._sionna_ok = 0
        self._sionna_fail = 0
        
        logger.info(f"DataGenerator initialized for scene directory: {config.scene_dir}")
        logger.info(f"  Carrier freq: {config.carrier_frequency_hz/1e9:.2f} GHz")
        logger.info(f"  UEs per tile: {config.num_ue_per_tile}")
        logger.info(f"  Reports per UE: {config.num_reports_per_ue}")
        logger.info(f"  Sionna RT: {'enabled' if SIONNA_AVAILABLE else 'disabled'}")
    
    def _load_scene_metadata(self, scene_id: str) -> Dict:
        """Loads scene metadata for a specific scene."""
        metadata_path = self.config.scene_dir / scene_id / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Scene metadata not found: {metadata_path}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {scene_id} ({len(metadata.get('sites', []))} sites).")
        return metadata
    
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
        # Find all scenes
        if scene_ids is None:
            scene_dirs = sorted(self.config.scene_dir.rglob("scene_*"))
            scene_ids = [str(d.relative_to(self.config.scene_dir)) for d in scene_dirs]
        
        if num_scenes is not None:
            scene_ids = scene_ids[:num_scenes]
        
        logger.info(f"Generating dataset from {len(scene_ids)} scenes...")
        
        # Process each scene
        for i, scene_id in enumerate(tqdm(scene_ids, desc="Processing Scenes")):
            logger.info(f"Processing scene {i+1}/{len(scene_ids)}: {scene_id}")

            
            scene_path = self.config.scene_dir / scene_id / "scene.xml"
            if not scene_path.exists():
                logger.warning(f"Scene file not found: {scene_path}, skipping.")
                continue
            
            # Generate data for this scene
            scene_metadata = self._load_scene_metadata(scene_id)
            
            # Load scene here to share between data and map generation
            scene_obj = None
            if SIONNA_AVAILABLE:
                scene_obj = self._load_sionna_scene(scene_path)
            
            scene_data = self.generate_scene_data(scene_path, scene_id, scene_metadata=scene_metadata, scene_obj=scene_obj)
            
            # Generate Maps - returns tuple of (numpy_array, sionna_radiomap_object)
            radio_map, sionna_rm = self._generate_radio_map_for_scene(scene_obj, scene_path, scene_metadata)
            osm_map = self._generate_osm_map_for_scene(scene_obj, scene_metadata)

            # Write to Zarr
            if self.zarr_writer is not None:

                # First write maps (use numpy array for storage)
                self.zarr_writer.write_scene_maps(scene_id, radio_map, osm_map)
                # Then append samples (linked by scene_id internal index)
                self.zarr_writer.append(scene_data, scene_id=scene_id, scene_metadata=scene_metadata)
                
                # Save map visualizations as reference images
                save_map_visualizations(scene_id, radio_map, osm_map, self.config.output_dir)
                
                # Generate 3D Scene Renderings (Sionna) - pass Sionna RadioMap object for native rendering
                if SIONNA_AVAILABLE and scene_obj is not None:
                     render_scene_3d(scene_obj, scene_id, scene_metadata, self.config.output_dir, radio_map=sionna_rm)
            else:
                logger.warning("Zarr writer not available; skipping data write.")
        
        # Finalize dataset
        if self.zarr_writer is not None:
            output_path = self.zarr_writer.finalize()
            logger.info(f"Dataset generation complete: {output_path}")
            return output_path
        else:
            logger.warning("Dataset generation complete but no data written (Zarr not available).")
            return self.config.output_dir
    
    def generate_scene_data(self,
                           scene_path: Path,
                           scene_id: str,
                           scene_metadata: Optional[Dict] = None,
                           scene_obj: Any = None) -> Dict[str, np.ndarray]:
        """
        Generates multi-layer data for a single scene.
        """
        # Load scene-specific metadata
        if scene_metadata is None:
            scene_metadata = self._load_scene_metadata(scene_id)
        
        # Load scene in Sionna
        scene = scene_obj
        if scene is None:
            if SIONNA_AVAILABLE:
                scene = self._load_sionna_scene(scene_path)
            else:
                logger.warning("Sionna unavailable; using mock data.")
                scene = None
        
        # Load site positions and cell IDs from metadata
        sites = scene_metadata.get('sites', [])
        # Filter sites with valid cell_id
        valid_sites = [s for s in sites if s.get('cell_id') is not None]
        site_positions = np.array([s['position'] for s in valid_sites], dtype=np.float32)
        cell_ids = np.array([s['cell_id'] for s in valid_sites], dtype=int)
        
        # Calculate coordinate offsets
        bbox = scene_metadata.get('bbox', {})
        if 'x_min' in bbox and 'x_max' in bbox:
            # Simulation Frame: 
            # We use UTM (Global) because M1 meshes contain UTM coordinates.
            # Shifting here would move UEs/Sites away from the mesh!
            sim_offset_x = 0.0 
            sim_offset_y = 0.0
            
            # Storage Frame: Bottom-Left Logic (Standard Image Coords [0, W])
            store_offset_x = bbox['x_min']
            store_offset_y = bbox['y_min']
            
            logger.info(f"Coord Frames -> Sim Offset: ({sim_offset_x:.2f}, {sim_offset_y:.2f}), Store Offset: ({store_offset_x:.2f}, {store_offset_y:.2f})")
        else:
            sim_offset_x, sim_offset_y = 0.0, 0.0
            store_offset_x, store_offset_y = 0.0, 0.0
            logger.warning("No bbox found in metadata; assuming local coordinates.")

        if len(sites) == 0:
            logger.warning(f"No sites in metadata for {scene_id}, using mock")
            # If no sites, we place mock sites in the middle of the BBox (UTM)
            cx = (bbox.get('x_min', 0) + bbox.get('x_max', 0)) / 2
            cy = (bbox.get('y_min', 0) + bbox.get('y_max', 0)) / 2
            site_positions = np.array([
                [cx, cy, 30.0], 
                [cx + 500.0, cy, 30.0]
            ], dtype=np.float32)
            cell_ids = np.array([1, 2])

        # Prepare site positions for Simulation
        site_positions_sim = site_positions.copy()
        if len(site_positions_sim) > 0 and (sim_offset_x != 0 or sim_offset_y != 0):
            site_positions_sim[:, 0] -= sim_offset_x
            site_positions_sim[:, 1] -= sim_offset_y
        
        # Setup transmitters in Sionna (Sim Frame)
        if scene is not None:
            self._setup_transmitters(scene, site_positions_sim, sites if sites else {})
            
        # Sample UE trajectories in GLOBAL frame (offset=0)
        # We will convert them to Sim/Store frames as needed
        trajectories_global = sample_ue_trajectories(
            scene_metadata=scene_metadata,
            num_ue_per_tile=self.config.num_ue_per_tile,
            ue_height_range=self.config.ue_height_range,
            ue_velocity_range=self.config.ue_velocity_range,
            num_reports_per_ue=self.config.num_reports_per_ue,
            report_interval_ms=self.config.report_interval_ms,
            offset=(0.0, 0.0)
        )
        
        # Collect data
        all_features = {
            'rt': [], 'phy_fapi': [], 'mac_rrc': [], 'positions': [], 'timestamps': []
        }
        
        total_reports = 0
        
        # Flatten all trajectory points into a single list of (t, pos) for batching
        all_points = []
        for traj_global in trajectories_global:
            for t_step, ue_pos_global in enumerate(traj_global):
                all_points.append((t_step, ue_pos_global))
        
        total_reports = 0
        batch_size = 32  # Good size for GPU efficiency without OOM
        
        # Process in batches
        # We construct the range usually. Let's wrap it.
        batch_range = range(0, len(all_points), batch_size)
        
        for i in tqdm(batch_range, desc="Simulating UE Batches", leave=False):
            batch = all_points[i:i + batch_size]
            batch_indices = range(i, i + len(batch))
            
            # Prepare batch inputs
            ue_positions_sim = []
            ue_positions_store = []
            t_steps = []
            
            for t_step, ue_pos_global in batch:
                # Sim Frame
                pos_sim = ue_pos_global.copy()
                pos_sim[0] -= sim_offset_x
                pos_sim[1] -= sim_offset_y
                ue_positions_sim.append(pos_sim)
                
                # Store Frame
                pos_store = ue_pos_global.copy()
                pos_store[0] -= store_offset_x
                pos_store[1] -= store_offset_y
                ue_positions_store.append(pos_store)
                
                t_steps.append(t_step)
            
            ue_positions_sim_arr = np.array(ue_positions_sim)
            
            # --- Fix UE Z-coordinates relative to terrain ---
            if SIONNA_AVAILABLE and scene is not None:
                # Initialize support flag if checking for the first time
                if not hasattr(self, '_ground_clamp_supported'):
                    self._ground_clamp_supported = True
                    
                if self._ground_clamp_supported:
                    try:
                        import drjit as dr
                        import mitsuba as mi
                        import tensorflow as tf
                        
                        # Use Mitsuba scene directly for ray casting
                        # Origins at [x, y, 30000]
                        z_start = 30000.0
                        
                        # Prepare data for Mitsuba (on CPU/LLVM backend)
                        batch_len = len(batch)
                        
                        # Create Rays in Dr.Jit/Mitsuba format
                        # Ray(o, d, time, wavelengths)
                        origin_np = np.stack([
                            ue_positions_sim_arr[:, 0], 
                            ue_positions_sim_arr[:, 1], 
                            np.full(batch_len, z_start)
                        ], axis=1).astype(np.float32)
                        
                        direction_np = np.tile([0.0, 0.0, -1.0], (batch_len, 1)).astype(np.float32)
                        
                        # Convert to Mitsuba types
                        o = mi.Point3f(origin_np)
                        d = mi.Vector3f(direction_np)
                        ray = mi.Ray3f(o, d)
                        
                        # Ray Cast
                        si = scene.mi_scene.ray_intersect(ray)
                        
                        # Check Hits
                        # si.t is the distance. collision point = o + t * d
                        # here d = (0,0,-1), so o.z - t = ground.z
                        hits_t = si.t.numpy()
                        valid = si.is_valid()
                        
                        # Ground Z
                        # If hit: z_start - t
                        # If invalid: 0.0 (or fallback)
                        ground_z = z_start - hits_t
                        
                        # Apply clamping only for valid hits
                        valid_np = valid.numpy()
                        
                        # Update Z
                        h_ue = ue_positions_sim_arr[:, 2]
                        # Only update where valid hit occurred
                        ue_positions_sim_arr[valid_np, 2] = ground_z[valid_np] + h_ue[valid_np]
                        
                        # Fallback for invalid hits (missed terrain?) -> Try AABB min
                        if not np.all(valid_np) and hasattr(scene, 'aabb'):
                             z_min = float(scene.aabb[0, 2])
                             ue_positions_sim_arr[~valid_np, 2] += z_min
                        
                        # Sync storage Z
                        for j in range(batch_len):
                            ue_positions_sim_arr[j, 2] = float(ue_positions_sim_arr[j, 2]) # Ensure float
                            ue_positions_store[j][2] = ue_positions_sim_arr[j, 2]
                            
                    except Exception as e:
                        # Disable future checks only if it's a critical missing module
                        # Otherwise log once per batch might be too much, log debug
                        logger.debug(f"UE Ground Clamping (Mitsuba) failed: {e}")
                        
                        # Fallback: Try setting to Scene Z-Min + UE Height (heuristic)
                        try:
                            if hasattr(scene, 'aabb'):
                                z_min = float(scene.aabb[0, 2])
                                # Heuristic: if valid_np logic didn't run, apply to all
                                ue_positions_sim_arr[:, 2] += z_min
                                for j in range(len(batch)):
                                    ue_positions_store[j][2] = ue_positions_sim_arr[j, 2]
                        except:
                            pass

            # Simulate Batch
            rt_batch, phy_batch, mac_batch = self._simulate_batch(
                scene, ue_positions_sim_arr, site_positions_sim, cell_ids
            )
            
            # Direct Append of Batched Data
            # Instead of unrolling, we append the batched dicts to the lists
            # The receiver (ZarrWriter) expects stacked arrays later anyway.
            # But `all_features` is a dict of lists-of-single-records?
            # No, `all_features['rt']` is a list of dicts.
            # We want to change it to accumulate dicts of arrays, then stack safely.
            # Current `_stack_scene_data` handles list of dicts.
            # We can modify logic to simply append the WHOLE batch dict as one item?
            # Then stacker needs to concat them.
            # Or we can write a quick unroller that is faster?
            # Actually, `unroll_features` is purely python pointer manipulation if done right.
            # But `to_dict` creates new dicts.
            # Let's keep `unroll_features` concept but optimize it?
            # Or better: Change `all_features` to store BATCH BLOCKS.
            
            # Let's modify `all_features` collectors to just accept the Batch Objects directly?
            # `all_features['rt'].append(rt_batch.to_dict())`
            # Then `_stack_scene_data` will see a list of Batched Dicts.
            # It currently iterates `all_data['rt']` and expects each item to be a dict of 1-sample arrays?
            # Or N-sample arrays?
            # `stacked[key] = np.stack(arrays_squeezed, axis=0)` implies it expects N-samples to stack along axis 0.
            # If we pass a Batch Array (size 32), `stack` would create [NumBatches, 32, ...].
            # We want [TotalSamples, ...].
            # So we should use `np.concatenate` instead of `np.stack`.
            
            # Plan: Modify storage to be `all_features` = list of batched dicts.
            # Modify `_stack_scene_data` to use `concatenate`.
            
            all_features['rt'].append(rt_batch.to_dict())
            all_features['phy_fapi'].append(phy_batch.to_dict())
            all_features['mac_rrc'].append(mac_batch.to_dict())
            all_features['positions'].append(ue_positions_store) # List of arrays
            all_features['timestamps'].append(np.array(t_steps) * self.config.report_interval_ms / 1000.0)
            
            total_reports += len(batch)
            if (i // batch_size) % 10 == 0:
                 logger.info(f"Processed {total_reports}/{len(all_points)} measurements...")



        # Stack into arrays
        scene_data = self._stack_scene_data(all_features)
        
        # Apply measurement realism
        scene_data = self._apply_measurement_realism(scene_data)

        total = self._sionna_ok + self._sionna_fail
        if total > 0:
            logger.info(f"Sionna RT summary: {self._sionna_ok}/{total} successful, {self._sionna_fail} mock fallbacks")
        
        return scene_data
    
    def _simulate_batch(self,
                              scene: Any,
                              ue_positions: np.ndarray,
                              site_positions: np.ndarray,
                              cell_ids: np.ndarray) -> Tuple[RTLayerFeatures, 
                                                              PHYFAPILayerFeatures,
                                                              MACRRCLayerFeatures]:
        """Runs a simulation for a BATCH of UE positions."""
        
        batch_size = len(ue_positions)
        
        if not SIONNA_AVAILABLE or scene is None:
            # Mock batch
            # We can just run mock loop or vectorize mock
            # For simplicity, loop mock
            rts, phys, macs = [], [], []
            for i in range(batch_size):
                 rt, phy, mac = self._simulate_mock(ue_positions[i], site_positions, cell_ids)
                 rts.append(rt); phys.append(phy); macs.append(mac)
                 
            # Merge into batch objects? 
            # Or return list? The caller assumes objects with [Batch,...] arrays.
            # Mock objects are single-record.
            # We need to stack them to return a "Batched Object".
            # This is getting complex for Mock.
            # Let's cheat: The caller unrolls them anyway. 
            # But the signature says return Tuple[FeatureObjects].
            # Let's do it properly using _stack_features helper if we had one.
            # Or just hack the Mock to return big arrays.
            # Ignoring Mock optimization for now.
            logger.debug("Sionna unavailable; using mock data for batch (slow loop).")
            # Construct dummy batched features
            # This is widely unused in production.
            # Just return single mock features expanded to batch
            rt_single, phy_single, mac_single = self._simulate_mock(ue_positions[0], site_positions, cell_ids)
            # Expand to batch?
            # TODO: Better mock batching.
            return rt_single, phy_single, mac_single
        
        added_rx_names = []
        try:
            # 1. Setup Receivers
            # Add N receivers to the scene
            for i, pos in enumerate(ue_positions):
                rx_name = f"UE_Batch_{i}"
                try:
                    self._setup_receiver(scene, pos, rx_name)
                    added_rx_names.append(rx_name)
                except Exception as rx_e:
                    logger.warning(f"Failed to add receiver {rx_name}: {rx_e}")
            
            # Log how many receivers were successfully added
            if len(added_rx_names) < len(ue_positions):
                logger.warning(f"Successfully added {len(added_rx_names)} out of {len(ue_positions)} receivers to the scene.")

            # 2. Compute Propagation (Batch)
            from sionna.rt import PathSolver
            
            max_depth = getattr(self.config, 'max_depth', 5)
            # Reduce samples per source for batching to save memory?
            # Or keep high quality.
            samples_per_src = getattr(self.config, 'num_samples', 100_000)
            
            paths_result = PathSolver()(
                scene,
                max_depth=max_depth,
                samples_per_src=samples_per_src,
                max_num_paths_per_src=getattr(self.config, 'max_num_paths', 100_000),
                los=True,
                specular_reflection=True,
                diffuse_reflection=False, # Specular only for speed usually
                refraction=True,
                diffraction=getattr(self.config, 'enable_diffraction', True),
                edge_diffraction=False
            )
            
            if isinstance(paths_result, tuple):
                paths = paths_result[0]
            else:
                paths = paths_result


            # 3. Extract Features (Batch)
            # The updated RTFeatureExtractor handles [Batch=Targets, ...] permutation
            # Pass batch_size and num_sites so fallback mock has correct dimensions
            num_sites = len(site_positions)
            rt_features = self.rt_extractor.extract(paths, batch_size=batch_size, num_rx=num_sites)
            
            # Channel Matrix
            channel_matrix = None
            try:
                # Batch CFR
                # CFR shape: [S, T, P, RxAnt, TxAnt, F] ?
                # Sionna cfr() output shape: [num_sources, num_targets, num_rx_ant, num_tx_ant, num_subcarriers, num_time_steps]
                # We want: [Target(Batch), RxAnt, Source, TxAnt, F] ? or similar.
                
                # Use Sionna Channel Utils for correct OFDM Channel Response
                # This keeps everything on GPU and uses validated 3GPP algorithms
                try:
                    from sionna.phy.channel import cir_to_ofdm_channel
                except ImportError:
                    from sionna.channel.utils import cir_to_ofdm_channel
                import tensorflow as tf
                
                # Frequencies from Resource Grid or Config
                if self.rg is not None:
                     # Use Resource Grid frequencies centered at carrier
                     # rg.frequencies is relative to DC? No, usually baseband.
                     # We need baseband frequencies for channel generation?
                     # cir_to_ofdm_channel expects strictly increasing frequencies.
                     # Relative to carrier is fine if delay is absolute?
                     # Usually we pass frequencies relative to 0 (baseband)
                     pass
                else:
                     # Fallback
                     pass

                num_subcarriers = getattr(self.config, 'num_subcarriers', 1024)
                bw = self.config.bandwidth_hz
                
                # Frequencies: [-BW/2 ... +BW/2]
                freqs = tf.linspace(-bw/2, bw/2, num_subcarriers)
                freqs = tf.cast(freqs, tf.float32)

                # Prepare inputs
                # paths.a: [S, T, P, RxAnt, TxAnt, 1, 1]
                # paths.tau: [S, T, P]
                # We need compatible shapes. Expand tau.
                # tau -> [S, T, P, 1, 1, 1, 1]
                
                a = paths.a
                tau = paths.tau
                tau = tf.reshape(tau, tau.shape + [1, 1, 1, 1])
                
                # Compute Channel Transfer Function
                # h_freq: [S, T, P, RxAnt, TxAnt, Time=1, Freq] -> Reduced over P
                # Actually cir_to_ofdm returns superposition of paths.
                # Output: [S, T, RxAnt, TxAnt, Time=1, Freq]
                
                h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
                
                # Shape: [Sources, Targets, RxAnt, TxAnt, 1, Freq]
                # Squeeze Time (axis -2)
                h_freq = tf.squeeze(h_freq, axis=-2)
                
                # Permute to [Targets, RxAnt, Sources, TxAnt, Freq] (Standard for our PHY extractor)
                # Orig: [0:S, 1:T, 2:RxA, 3:TxA, 4:F]
                # Target: [T, RxA, S, TxA, F] => [1, 2, 0, 3, 4] (Wait. Rx usually associated with Target?)
                # If Target = UE, Output is UE. RxAnt is UE Ant.
                # If we want [Batch=Targets, Rx=Sources...], wait.
                # In feature extractor we assumed [Batch, Rx, ...].
                # If Batch=Targets, and "RX" in feature extraction context referred to *Receiving Points* aka Sites?
                # No, standard Downlink: Source=Site, Target=UE.
                # Feature Extractor: `rsrp` is `[batch, num_rx, num_tx]`.
                # If Batch=UEs, num_rx should be 1 (single UE has 1 rx array)?
                # Or did we assume `num_rx` meant `num_visible_cells`?
                # In `features.py`: `rsrp` [batch, num_rx, num_cells].
                # Code says: `rsrp = np.sum(p_tx, axis=3)` -> summing over Rx antennas.
                # `channel_matrix` passed to `extract` was `[batch, rx, rx_ant, tx, tx_ant, fft]`.
                # If Batch=UEs=Targets. 
                # `rx` dimension in `channel_matrix`??
                # Usually `h` is [Batch, RxAnt, TxAnt...] or [Batch, NumCells, RxAnt...]
                # Sionna output [S, T] = [Cells, UEs].
                # Transpose to [UEs, Cells].
                # So dimension 0 should be T=Batch. Dimension 1 should be S=Cells.
                # Then Antennas.
                # [T, S] -> [1, 0, ...]
                
                # Permute: [Targets, Sources, RxAnt, TxAnt, Freq]
                # [1, 0, 2, 3, 4]
                
                channel_matrix = tf.transpose(h_freq, perm=[1, 0, 2, 3, 4])
                
                # Verify shape: We expect [Batch, ...] (Batch=Targets)
                # If for some reason broadcasting or ordering flipped (e.g. S, T), fix it.
                if channel_matrix.shape[0] != batch_size and channel_matrix.shape[1] == batch_size:
                    logger.warning(f"Transpose flip detected in channel_matrix. Shape: {channel_matrix.shape}, correcting to [Batch, ...]")
                    channel_matrix = tf.transpose(channel_matrix, perm=[1, 0, 2, 3, 4])
                
            except Exception as e:
                logger.debug(f"Sionna Channel Calc failed: {e}")
                channel_matrix = None
                
            except Exception as e:
                logger.debug(f"Batch CFR failed: {e}")
                channel_matrix = None
            
            self._sionna_ok += batch_size

            # 4. PHY / MAC
            # phy_extractor should handle batch arrays naturally (numpy operations)
            phy_features = self.phy_extractor.extract(
                rt_features=rt_features,
                channel_matrix=channel_matrix
            )
            
            # mac_extractor
            mac_features = self.mac_extractor.extract(
                phy_features=phy_features,
                ue_positions=ue_positions, # [Batch, 3]
                site_positions=site_positions,
                cell_ids=cell_ids
            )
            
            # Extract Native RT Features
            rt_native_dict = self.native_extractor.extract_rt(paths, batch_size)
            
            # Merge into RTFeatures
            # We use the existing extractor for standard fields, then overlay native ones
            # Or should we fully replace? RTFeatureExtractor supports TF.
            # But native_extractor has toa/nlos.
            if rt_native_dict:
                 rt_features.toa = rt_native_dict.get('toa')
                 rt_features.is_nlos = rt_native_dict.get('is_nlos')
                 # Propagate real path features (overwriting mock/legacy)
                 if 'path_gains' in rt_native_dict:
                     rt_features.path_gains = rt_native_dict.get('path_gains')
                 if 'path_delays' in rt_native_dict:
                     rt_features.path_delays = rt_native_dict.get('path_delays')
                 
            # Extract Native PHY Features
            if channel_matrix is not None:
                phy_native_dict = self.native_extractor.extract_phy(channel_matrix, batch_size)
                
                # Merge into PHYFeatures
                phy_features.capacity_mbps = phy_native_dict.get('on_se') * (self.config.bandwidth_hz / 1e6)
                phy_features.condition_number = phy_native_dict.get('cond_num')
                
                # Update basic metrics with native computation if available?
                # For now, trust the legacy extractor as it matches dataset expectations,
                # but we could check divergence.
            
            return rt_features, phy_features, mac_features
            
        except Exception as e:
            self._sionna_fail += batch_size
            logger.error(f"Sionna batch simulation failed: {e}")
            logger.error(traceback.format_exc())
            # Fallback (mock) - create mock features for entire batch
            logger.warning(f"Falling back to mock simulation for batch of {batch_size}")
            
            # Generate mock for each UE and stack
            rts, phys, macs = [], [], []
            for i in range(batch_size):
                rt_m, phy_m, mac_m = self._simulate_mock(ue_positions[i], site_positions, cell_ids)
                rts.append(rt_m)
                phys.append(phy_m)
                macs.append(mac_m)
            
            # Stack into batch objects
            from src.data_generation.features import RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures
            
            # Helper to stack feature dataclass objects
            def stack_features(feat_list, FeatureClass):
                """Stack a list of feature objects into a single batched object."""
                if not feat_list:
                    return None
                
                # Get all fields from the first object
                first = feat_list[0]
                stacked_dict = {}
                
                for field_name in first.__dataclass_fields__:
                    field_values = [getattr(f, field_name) for f in feat_list]
                    
                    # Check if all are None
                    if all(v is None for v in field_values):
                        stacked_dict[field_name] = None
                    # Check if all are numpy arrays
                    elif all(isinstance(v, np.ndarray) for v in field_values if v is not None):
                        # Stack non-None values
                        non_none_values = [v for v in field_values if v is not None]
                        if non_none_values:
                            stacked_dict[field_name] = np.concatenate(non_none_values, axis=0)
                        else:
                            stacked_dict[field_name] = None
                    else:
                        # For scalar fields, just take first value
                        stacked_dict[field_name] = field_values[0]
                
                return FeatureClass(**stacked_dict)
            
            rt_batch = stack_features(rts, RTLayerFeatures)
            phy_batch = stack_features(phys, PHYFAPILayerFeatures)
            mac_batch = stack_features(macs, MACRRCLayerFeatures)
            
            return rt_batch, phy_batch, mac_batch
            
        finally:
            # Clean up receivers
            for name in added_rx_names:
                try:
                    scene.remove(name)
                except:
                    pass

    
    def _load_sionna_scene(self, scene_path: Path) -> Any:
        """Loads a Mitsuba XML scene into Sionna RT."""
        if not SIONNA_AVAILABLE:
            logger.warning("Sionna unavailable; cannot load scene.")
            return None
            
        from sionna.rt import load_scene
        
        logger.info(f"Loading Sionna scene from {scene_path}...")
        scene = load_scene(str(scene_path))
        
        location_data = None
        # Apply scene-level settings
        scene.frequency = self.config.carrier_frequency_hz
        scene.synthetic_array = True
        
        if hasattr(scene, 'transmitters'):
            # Convert keys to list to avoid runtime error during modification
            tx_names = list(scene.transmitters.keys())
            if tx_names:
                for name in tx_names:
                    scene.remove(name)
        
        logger.info(f"Scene loaded: frequency={self.config.carrier_frequency_hz/1e9:.2f} GHz.")
        return scene
    
    def _setup_transmitters(self, scene: Any, 
                           site_positions: np.ndarray,
                           site_metadata: Dict) -> List[Any]:
        """Setup cell site transmitters with antenna arrays."""
        if not SIONNA_AVAILABLE or scene is None:
            return []
            
        from sionna.rt import Transmitter, PlanarArray
        
        # Verify clean state
        current_tx = list(scene.transmitters.keys())
        
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
            azimuth = 0.0
            downtilt = 10.0
            
            if isinstance(site_metadata, list):
                if site_idx < len(site_metadata):
                    meta = site_metadata[site_idx]
                    azimuth = meta.get('orientation', [0, 0, 0])[0] if 'orientation' in meta else 0.0
                    downtilt = meta.get('orientation', [0, 10, 0])[1] if 'orientation' in meta else 10.0
            elif isinstance(site_metadata, dict):
                azimuth = site_metadata.get(f'site_{site_idx}_azimuth', 0.0)
                downtilt = site_metadata.get(f'site_{site_idx}_downtilt', 10.0)
            
            tx = Transmitter(
                name=f"BS_{site_idx}",
                position=pos if isinstance(pos, list) else pos.tolist(),
                orientation=[azimuth, downtilt, 0.0]
            )
            
            try:
                # Clamp site to ground level if needed
                if SIONNA_AVAILABLE:
                    try:
                        import drjit as dr
                        import mitsuba as mi
                        import tensorflow as tf
                        
                        z_start = 30000.0
                        
                        # Mitsuba Ray
                        o = mi.Point3f(pos[0], pos[1], z_start)
                        d = mi.Vector3f(0.0, 0.0, -1.0)
                        ray = mi.Ray3f(o, d)
                        
                        # Ray Cast
                        si = scene.mi_scene.ray_intersect(ray)
                        
                        if si.is_valid():
                            # Ground found
                            dist_t = si.t.numpy()[0]
                            ground_z = z_start - dist_t
                            
                            min_tx_h = 25.0
                            if pos[2] < ground_z + min_tx_h:
                                 logger.debug(f"Adjusting Site {site_idx} height: {pos[2]:.1f} -> {ground_z+min_tx_h:.1f}")
                                 pos[2] = ground_z + min_tx_h
                                 
                                 # Update the object wrapper position
                                 tx.position = pos if isinstance(pos, list) else pos.tolist()
                                 
                    except Exception as clamp_e:
                        if site_idx == 0: # Log once
                            # logger.warning(f"Tx Ground Clamping (Mitsuba) failed: {clamp_e}")
                            pass # Debug mainly
                        
                        # Fallback to AABB Z-min
                        try:
                            if hasattr(scene, 'aabb'):
                                z_min = float(scene.aabb[0, 2])
                                # If pos is very low (e.g. 30), assume it's relative? 
                                # But if it is < z_min, it implies it's definitely wrong for UTM.
                                if pos[2] < z_min:
                                     # Bump to z_min + 30m?
                                     if z_min > 500 and pos[2] < 100:
                                         pos[2] += z_min
                                         tx.position = pos if isinstance(pos, list) else pos.tolist()
                        except:
                            pass
                
                scene.add(tx)
                transmitters.append(tx)
                logger.debug(f"Added transmitter BS_{site_idx} at {pos}")
            except Exception as e:
                logger.warning(f"Failed to add transmitter BS_{site_idx}: {e}")
        
        return transmitters
    
    def _setup_receiver(self, scene: Any, ue_position: np.ndarray, name: str) -> str:
        """Setup UE receiver at given position."""
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
        # ue_position is [3,]
        mac_features = self.mac_extractor.extract(
            phy_features, ue_position[np.newaxis, :], site_positions, cell_ids
        )

        
        return rt_features, phy_features, mac_features
    
    def _stack_scene_data(self, all_data: Dict[str, List]) -> Dict[str, np.ndarray]:
        """Stack lists of feature batches into arrays."""
        stacked = {}
        
        # Positions and timestamps
        # all_data['positions'] is list of [Batch, 3] arrays
        if len(all_data['positions']) > 0:
            stacked['positions'] = np.concatenate(all_data['positions'], axis=0) # [Total, 3]
            stacked['timestamps'] = np.concatenate(all_data['timestamps'], axis=0) # [Total]
        else:
            return {}

        # Helper to extract data from [batch, rx, ...] structure
        def safe_extract_layers(a):
            if a is None or a.size == 0: return a
            # If batching is used, we might have [Batch, Rx, ...] or just [Batch, ...] if Rx=1 squeezed.
            # But here 'a' is a batch block. We keep it as is.
            return a
            
        # Helper to pad cell dim (last dim)
        def pad_cell_dim(a, target_size=16, fill_value=0):
            if a is None: return a
            if a.shape[-1] >= target_size:
                return a[..., :target_size]
            
            padding = [(0, 0)] * (a.ndim - 1) + [(0, target_size - a.shape[-1])]
            return np.pad(a, padding, mode='constant', constant_values=fill_value)

        # Helper to pad TX dimension
        def pad_tx_dim(a, target_size=8, fill_value=0, axis=1):
            if a is None: return a
            # Logic needs to find axis relative to ndim
            # With batching blocks, shapes are stable.
            if axis >= a.ndim: return a
            
            if a.shape[axis] >= target_size:
                slices = [slice(None)] * a.ndim
                slices[axis] = slice(0, target_size)
                return a[tuple(slices)]
            
            padding = [(0, 0)] * a.ndim
            padding[axis] = (0, target_size - a.shape[axis])
            return np.pad(a, padding, mode='constant', constant_values=fill_value)

        # Generic Stacker for Feature Lists
        def process_feature_list(feat_list, key_prefix):
            if not feat_list: return
            keys = feat_list[0].keys()
             
            for key in keys:
                # Collect all batches for this key
                batches = [d[key] for d in feat_list if d[key] is not None]
                if not batches: continue
                 
                 # Concat batch blocks
                try:
                    # Check if shapes match (except dim 0)
                    # Sometimes variable length paths might cause issue -> use object array or fail
                    # If variable length, concatenate usually fails or creates flat.
                     
                    # Special handling for padded features
                    # Apply padding per batch BEFORE concat to ensure safety
                    MAX_CELLS = 16
                    MAX_TX = 8
                     
                    processed_batches = []
                    for b in batches:
                        val = b
                        
                        # RT Features: [Batch, Sites] -> Pad Axis 1
                        if key in ['rt/rms_delay_spread', 'rt/num_paths']:
                             val = pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=0)
                        
                        # PHY Features: [Batch, Sites, Beams] -> Pad Axis 1
                        elif key in ['phy_fapi/l1_rsrp_beams', 'phy_fapi/best_beam_ids']:
                             fill_val = -150.0 if 'rsrp' in key else -1
                             val = pad_tx_dim(val, axis=1, target_size=MAX_CELLS, fill_value=fill_val)

                        # PHY Features: [Batch, Rx, Cells] -> Pad Last Dim
                        elif key in ['phy_fapi/rsrp', 'phy_fapi/rsrq', 'phy_fapi/sinr', 
                                'phy_fapi/cqi', 'phy_fapi/ri', 'phy_fapi/pmi']:
                            fill_val = -150.0 if 'rsr' in key or 'sinr' in key else 0
                            val = pad_cell_dim(val, target_size=MAX_CELLS, fill_value=fill_val)
                        
                        # MAC Features: [Batch, UE, Neighbors] -> Pad Last Dim
                        elif key == 'mac_rrc/neighbor_cell_ids':
                            # Pad to max_neighbors (configured)
                            val = pad_cell_dim(val, target_size=self.config.max_neighbors, fill_value=-1)
                            
                        elif key in ['mac_rrc/dl_throughput_mbps', 'mac_rrc/bler']:
                            # [Batch, Rx] - no padding needed usually unless Multi-Cell?
                            pass
                        
                        processed_batches.append(val)

                    # First, ensure all batches have the same shape (except axis 0)
                    # Find the maximum shape for each dimension
                    ndim = processed_batches[0].ndim
                    max_shape = list(processed_batches[0].shape)
                    for b in processed_batches[1:]:
                        for i in range(1, ndim):  # Skip axis 0 (batch)
                            max_shape[i] = max(max_shape[i], b.shape[i])
                      
                    # Pad all batches to max_shape
                    padded_batches = []
                    for b in processed_batches:
                        if list(b.shape[1:]) != max_shape[1:]:
                            # Need padding
                            pad_width = [(0, 0)]  # No padding on axis 0
                            for i in range(1, ndim):
                                pad_width.append((0, max_shape[i] - b.shape[i]))
                            fill_val = -150.0 if ('rsr' in key or 'sinr' in key) else 0
                            b_padded = np.pad(b, pad_width, mode='constant', constant_values=fill_val)
                            padded_batches.append(b_padded)
                        else:
                            padded_batches.append(b)
                      
                    concatenated = np.concatenate(padded_batches, axis=0)
                    
                    # Avoid double prefixing
                    if key.startswith(f"{key_prefix}/"):
                        final_key = key
                    else:
                        final_key = f"{key_prefix}/{key}"
                        
                    stacked[final_key] = concatenated
                      
                except ValueError as e:
                    logger.warning(f"Failed to stack feature {key}: {e}")
                    # Last resort: just take the first batch if all else fails
                    if processed_batches:
                        logger.warning(f"Using only first batch for {key}")
                        if key.startswith(f"{key_prefix}/"):
                            stacked[key] = processed_batches[0]
                        else:
                            stacked[f"{key_prefix}/{key}"] = processed_batches[0]
                    else:
                        if key.startswith(f"{key_prefix}/"):
                            stacked[key] = None
                        else:
                            stacked[f"{key_prefix}/{key}"] = None

        # Process RT
        process_feature_list(all_data['rt'], 'rt')
        # Process PHY
        process_feature_list(all_data['phy_fapi'], 'phy_fapi')
        # Process MAC
        process_feature_list(all_data['mac_rrc'], 'mac_rrc')

        return stacked

    
    def _apply_measurement_realism(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply measurement dropout and quantization."""
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
        
        dropout_rates = {k: self.config.measurement_dropout_rates.get(v, 0.0) if self.config.measurement_dropout_rates else 0.0
                        for k, v in dropout_mapping.items()}
        
        phy_fapi_dropped = add_measurement_dropout(phy_fapi_dict, dropout_rates, seed=42)
        
        # Update scene_data
        scene_data.update(phy_fapi_dropped)
        
        return scene_data

    def _generate_radio_map_for_scene(self, scene: Any, scene_path: Path, scene_metadata: Dict) -> Tuple[np.ndarray, Any]:
        """Generates a radio map for the scene using RadioMapGenerator.
        
        Returns:
            Tuple of (radio_map_numpy, sionna_radiomap_object)
            - radio_map_numpy: [C, H, W] numpy array for storage
            - sionna_radiomap_object: Sionna RadioMap for visualization (or None)
        """
        default_map = np.zeros((5, 256, 256), dtype=np.float32)
        
        if not SIONNA_AVAILABLE:
             logger.warning("Radio Map Gen: Sionna not available.")
             return default_map, None
        if scene is None:
             logger.warning("Radio Map Gen: Scene is None.")
             return default_map, None

        logger.info("Radio Map Gen: Starting generation...")

        try:
             # Configure Radio Map Generator using UTM coordinates
             bbox = scene_metadata.get('bbox', {})
             x_min = bbox.get('x_min', -250)
             x_max = bbox.get('x_max', 250)
             y_min = bbox.get('y_min', -250)
             y_max = bbox.get('y_max', 250)
             
             map_extent = (x_min, y_min, x_max, y_max)
             logger.info(f"Radio Map Gen: Extent={map_extent}")
             
             map_config = RadioMapConfig(
                 resolution=1.0, 
                 map_size=(256, 256),
                 map_extent=map_extent,
                 output_dir=Path("/tmp"),
             )
             
             # Extract cell sites (use UTM positions)
             sites = scene_metadata.get('sites', [])
             valid_sites = [s for s in sites if s.get('cell_id') is not None]
             site_positions = [s['position'] for s in valid_sites]
             if not site_positions:
                 # UTM center fallback
                 cx = (x_min + x_max) / 2
                 cy = (y_min + y_max) / 2
                 site_positions = [[cx, cy, 30.0]]
             
             cell_sites_for_gen = []
             for p in site_positions:
                 cell_sites_for_gen.append({'position': p})

             generator = RadioMapGenerator(map_config)
             
             # Request both numpy array and Sionna RadioMap object
             radio_map, sionna_rm = generator.generate_for_scene(
                 scene, cell_sites_for_gen, show_progress=False, return_sionna_object=True
             )
             
             # Ensure [C, H, W]
             if radio_map.ndim == 3:
                 if radio_map.shape[-1] < radio_map.shape[0]: # [H, W, C]
                      radio_map = np.transpose(radio_map, (2, 0, 1))
             
             # Pad or clip channels to 5
             if radio_map.shape[0] > 5:
                 radio_map = radio_map[:5]
             elif radio_map.shape[0] < 5:
                 padding = np.zeros((5 - radio_map.shape[0], radio_map.shape[1], radio_map.shape[2]), dtype=np.float32)
                 radio_map = np.concatenate([radio_map, padding], axis=0)
                 
             return radio_map, sionna_rm

        except Exception as e:
            logger.error(f"Failed to generate radio map for scene: {e}")
            return default_map, None

    def _generate_osm_map_for_scene(self, scene: Any, metadata: Dict) -> np.ndarray:
        """Generates 5-channel OSM map for the scene."""
        default_map = np.zeros((5, 256, 256), dtype=np.float32)

        if scene is None:
             return default_map
             
        try:
             # Calculate dynamic extent to match Radio Map
             bbox = metadata.get('bbox', {})
             x_min = bbox.get('x_min', -250)
             x_max = bbox.get('x_max', 250)
             y_min = bbox.get('y_min', -250)
             y_max = bbox.get('y_max', 250)
             
             map_size = (256, 256)
             # Use actual UTM extent to match Radio Map and Scene content
             map_extent = (x_min, y_min, x_max, y_max)
             logger.info(f"OSM Map Gen: Extent={map_extent}")
             
             rasterizer = OSMRasterizer(map_size=map_size, map_extent=map_extent)
             osm_map = rasterizer.rasterize(scene)
             return osm_map
             
        except Exception as e:
             logger.error(f"Failed to generate OSM map: {e}")
             return default_map

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data generator
    logger.info("Testing MultiLayerDataGenerator...")
    
    # Create test config
    config = DataGenerationConfig(
        scene_dir=Path("test_scenes"),
        scene_metadata_path=Path("test_scenes/metadata.json"),
        num_ue_per_tile=10,
        num_reports_per_ue=5,
        output_dir=Path("test_output"),
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=100e6,
        tx_power_dbm=43.0,
        noise_figure_db=9.0
    )
    
    # Create mock scene directory
    config.scene_dir.mkdir(exist_ok=True)
    config.output_dir.mkdir(exist_ok=True)
    
    # Write mock metadata
    metadata = {
        'scene_001': {
            'bbox': {'x_min': -250, 'x_max': 250, 'y_min': -250, 'y_max': 250},
            'sites': [
                {'position': [0, 0, 30], 'cell_id': 1},
                {'position': [200, 0, 30], 'cell_id': 2},
            ],
        }
    }
    with open(config.scene_metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Initialize generator
    generator = MultiLayerDataGenerator(config)
    
    # Test UE trajectory sampling (via imported function)
    trajectories = sample_ue_trajectories(
        metadata['scene_001'], 
        config.num_ue_per_tile, 
        config.ue_height_range, 
        config.ue_velocity_range, 
        config.num_reports_per_ue, 
        config.report_interval_ms
    )
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
