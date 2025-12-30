"""
Batch simulation for ray tracing and feature extraction.
"""

import logging
import numpy as np
import traceback
from typing import Tuple, List, Any, Optional

logger = logging.getLogger(__name__)

# Try importing Sionna
try:
    import sionna
    from sionna.rt import PathSolver
    import tensorflow as tf
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    logger.warning("Sionna not available; BatchSimulator will operate in mock mode.")


class BatchSimulator:
    """
    Handles batch ray tracing simulations and feature extraction.
    
    Responsibilities:
    - Run batch ray tracing simulations
    - Extract RT/PHY/MAC features from simulation results
    - Handle channel matrix computation
    - Provide fallback mock simulation
    """
    
    def __init__(
        self,
        rt_extractor: Any,
        phy_extractor: Any,
        mac_extractor: Any,
        native_extractor: Any,
        config: Any,
        resource_grid: Optional[Any] = None
    ):
        """
        Initialize batch simulator.
        
        Args:
            rt_extractor: RT feature extractor instance
            phy_extractor: PHY/FAPI feature extractor instance
            mac_extractor: MAC/RRC feature extractor instance
            native_extractor: Sionna native KPI extractor instance
            config: Data generation configuration
            resource_grid: Optional Sionna resource grid for OFDM
        """
        self.rt_extractor = rt_extractor
        self.phy_extractor = phy_extractor
        self.mac_extractor = mac_extractor
        self.native_extractor = native_extractor
        self.config = config
        self.rg = resource_grid
        
        # Statistics
        self._sionna_ok = 0
        self._sionna_fail = 0
        
    def simulate_batch(
        self,
        scene: Any,
        ue_positions: np.ndarray,
        site_positions: np.ndarray,
        cell_ids: np.ndarray,
        transmitter_setup: Any
    ) -> Tuple[Any, Any, Any]:
        """
        Run simulation for a batch of UE positions.
        
        Args:
            scene: Sionna scene object
            ue_positions: Array of UE positions [N, 3]
            site_positions: Array of site positions [M, 3]
            cell_ids: Array of cell IDs
            transmitter_setup: TransmitterSetup instance for receiver setup
            
        Returns:
            Tuple of (RT features, PHY features, MAC features)
        """
        batch_size = len(ue_positions)
        
        if not SIONNA_AVAILABLE or scene is None:
            return self._simulate_mock_batch(ue_positions, site_positions, cell_ids)
        
        added_rx_names = []
        try:
            # 1. Setup Receivers
            for i, pos in enumerate(ue_positions):
                rx_name = f"UE_Batch_{i}"
                try:
                    transmitter_setup.setup_receiver(scene, pos, rx_name)
                    added_rx_names.append(rx_name)
                except Exception as rx_e:
                    logger.warning(f"Failed to add receiver {rx_name}: {rx_e}")
            
            if len(added_rx_names) < len(ue_positions):
                logger.warning(f"Successfully added {len(added_rx_names)} out of {len(ue_positions)} receivers.")

            # 2. Compute Propagation (Batch)
            paths = self._run_path_solver(scene)
            
            # 3. Extract RT Features
            num_sites = len(site_positions)
            rt_features = self.rt_extractor.extract(paths, batch_size=batch_size, num_rx=num_sites)
            
            # 4. Compute Channel Matrix
            channel_matrix = self._compute_channel_matrix(paths, batch_size)
            
            self._sionna_ok += batch_size

            # 5. Extract PHY Features
            phy_features = self.phy_extractor.extract(
                rt_features=rt_features,
                channel_matrix=channel_matrix
            )
            
            # 6. Extract MAC Features
            mac_features = self.mac_extractor.extract(
                phy_features=phy_features,
                ue_positions=ue_positions,
                site_positions=site_positions,
                cell_ids=cell_ids
            )
            
            # 7. Merge Native Features
            self._merge_native_features(paths, channel_matrix, batch_size, rt_features, phy_features)
            
            return rt_features, phy_features, mac_features
            
        except Exception as e:
            self._sionna_fail += batch_size
            logger.error(f"Sionna batch simulation failed: {e}")
            logger.error(traceback.format_exc())
            logger.warning(f"Falling back to mock simulation for batch of {batch_size}")
            
            return self._simulate_mock_batch(ue_positions, site_positions, cell_ids)
            
        finally:
            # Clean up receivers
            for name in added_rx_names:
                try:
                    scene.remove(name)
                except:
                    pass
    
    def _run_path_solver(self, scene: Any) -> Any:
        """Run Sionna path solver."""
        max_depth = getattr(self.config, 'max_depth', 5)
        samples_per_src = getattr(self.config, 'num_samples', 100_000)
        
        paths_result = PathSolver()(
            scene,
            max_depth=max_depth,
            samples_per_src=samples_per_src,
            max_num_paths_per_src=getattr(self.config, 'max_num_paths', 100_000),
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=True,
            diffraction=getattr(self.config, 'enable_diffraction', True),
            edge_diffraction=False
        )
        
        if isinstance(paths_result, tuple):
            return paths_result[0]
        return paths_result
    
    def _compute_channel_matrix(self, paths: Any, batch_size: int) -> Optional[Any]:
        """Compute OFDM channel matrix from paths."""
        try:
            from sionna.phy.channel import cir_to_ofdm_channel
        except ImportError:
            try:
                from sionna.channel.utils import cir_to_ofdm_channel
            except ImportError:
                logger.debug("cir_to_ofdm_channel not available")
                return None
        
        try:
            num_subcarriers = getattr(self.config, 'num_subcarriers', 1024)
            bw = self.config.bandwidth_hz
            
            # Frequencies: [-BW/2 ... +BW/2]
            freqs = tf.linspace(-bw/2, bw/2, num_subcarriers)
            freqs = tf.cast(freqs, tf.float32)

            # Prepare inputs
            a = paths.a
            tau = paths.tau
            tau = tf.reshape(tau, tau.shape + [1, 1, 1, 1])
            
            # Compute Channel Transfer Function
            h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
            
            # Shape: [Sources, Targets, RxAnt, TxAnt, 1, Freq]
            # Squeeze Time dimension
            h_freq = tf.squeeze(h_freq, axis=-2)
            
            # Permute to [Targets, RxAnt, Sources, TxAnt, Freq]
            channel_matrix = tf.transpose(h_freq, perm=[1, 2, 0, 3, 4])
            # Expand Rx dim at 1
            channel_matrix = tf.expand_dims(channel_matrix, 1)
            
            # Verify shape
            if channel_matrix.shape[0] != batch_size and channel_matrix.shape[1] == batch_size:
                logger.warning(f"Transpose flip detected in channel_matrix. Correcting to [Batch, ...]")
                channel_matrix = tf.transpose(channel_matrix, perm=[1, 0, 2, 3, 4])
            
            return channel_matrix
            
        except Exception as e:
            logger.debug(f"Channel matrix computation failed: {e}")
            return None
    
    def _merge_native_features(
        self,
        paths: Any,
        channel_matrix: Optional[Any],
        batch_size: int,
        rt_features: Any,
        phy_features: Any
    ) -> None:
        """Merge native Sionna features into extracted features."""
        # Extract Native RT Features
        rt_native_dict = self.native_extractor.extract_rt(paths, batch_size)
        
        if rt_native_dict:
            rt_features.toa = rt_native_dict.get('toa')
            rt_features.is_nlos = rt_native_dict.get('is_nlos')
            if 'path_gains' in rt_native_dict:
                rt_features.path_gains = rt_native_dict.get('path_gains')
            if 'path_delays' in rt_native_dict:
                rt_features.path_delays = rt_native_dict.get('path_delays')
        
        # Extract Native PHY Features
        if channel_matrix is not None:
            phy_native_dict = self.native_extractor.extract_phy(channel_matrix, batch_size)
            
            phy_features.capacity_mbps = phy_native_dict.get('on_se') * (self.config.bandwidth_hz / 1e6)
            phy_features.condition_number = phy_native_dict.get('cond_num')
    
    def _simulate_mock_batch(
        self,
        ue_positions: np.ndarray,
        site_positions: np.ndarray,
        cell_ids: np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """
        Fallback mock simulation for batch.
        
        This is a simplified implementation that should be expanded
        based on the actual mock simulation logic.
        """
        from ..features import RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures
        
        batch_size = len(ue_positions)
        logger.debug(f"Using mock simulation for batch of {batch_size}")
        
        # Generate mock features for each UE and stack
        rts, phys, macs = [], [], []
        for i in range(batch_size):
            # This would call the actual mock simulation method
            # For now, create minimal mock features
            rt_m = RTLayerFeatures(
                path_gains=np.random.randn(1, 10, 64) * 1e-8,
                path_delays=np.random.rand(1, 10, 64) * 1e-6,
                rms_delay_spread=np.random.rand(1) * 1e-7,
                k_factor=np.random.rand(1) * 10,
                num_paths=np.ones(1, dtype=np.int32) * 10
            )
            phy_m = PHYFAPILayerFeatures(
                rsrp=np.random.randn(1, 16) * 10 - 100,
                rsrq=np.random.randn(1, 16) * 5 - 10,
                sinr=np.random.randn(1, 16) * 10
            )
            mac_m = MACRRCLayerFeatures(
                serving_cell_id=np.zeros(1, dtype=np.int32),
                timing_advance=np.random.randint(0, 100, size=1)
            )
            rts.append(rt_m)
            phys.append(phy_m)
            macs.append(mac_m)
        
        # Stack into batch objects
        rt_batch = self._stack_features(rts, RTLayerFeatures)
        phy_batch = self._stack_features(phys, PHYFAPILayerFeatures)
        mac_batch = self._stack_features(macs, MACRRCLayerFeatures)
        
        return rt_batch, phy_batch, mac_batch
    
    def _stack_features(self, feat_list: List[Any], FeatureClass: type) -> Any:
        """Stack a list of feature objects into a single batched object."""
        if not feat_list:
            return None
        
        first = feat_list[0]
        stacked_dict = {}
        
        for field_name in first.__dataclass_fields__:
            field_values = [getattr(f, field_name) for f in feat_list]
            
            # Check if all are None
            if all(v is None for v in field_values):
                stacked_dict[field_name] = None
            # Check if all are numpy arrays
            elif all(isinstance(v, np.ndarray) for v in field_values if v is not None):
                non_none_values = [v for v in field_values if v is not None]
                if non_none_values:
                    stacked_dict[field_name] = np.concatenate(non_none_values, axis=0)
                else:
                    stacked_dict[field_name] = None
            else:
                # For scalar fields, just take first value
                stacked_dict[field_name] = field_values[0]
        
        return FeatureClass(**stacked_dict)
    
    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return {
            'sionna_ok': self._sionna_ok,
            'sionna_fail': self._sionna_fail
        }
