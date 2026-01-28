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
        self.use_mock_mode = getattr(config, 'use_mock_mode', False)
        self.allow_mock_fallback = getattr(config, 'allow_mock_fallback', True)
        self.require_sionna = getattr(config, 'require_sionna', False)
        self.require_cfr = getattr(config, 'require_cfr', False)
        
        # Statistics
        self._sionna_ok = 0
        self._sionna_fail = 0
        self._channel_matrix_logged = False
        self._rt_features_logged = False
        self._rt_merge_logged = False
        self._channel_matrix_fail = 0
        self._cfr_missing = 0
        self._diagnostic_logs = 0
        self._diagnostic_max = getattr(config, "rt_diagnostics_max", 10)
        self._channel_fail_log_every = getattr(config, "rt_fail_log_every", 100)
        self._channel_fail_log_count = 0
        
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
        
        if self.use_mock_mode:
            return self._simulate_mock_batch(ue_positions, site_positions, cell_ids)

        if not SIONNA_AVAILABLE or scene is None:
            if self.require_sionna or not self.allow_mock_fallback:
                raise RuntimeError("Sionna unavailable or scene is None while mock fallback is disabled.")
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
            if not self._channel_matrix_logged:
                try:
                    logger.info(f"Paths tau shape: {paths.tau.shape}")
                except Exception:
                    logger.info("Paths tau shape: <unavailable>")
            
            # 3. Extract RT Features
            num_sites = len(site_positions)
            rt_features = self.rt_extractor.extract(paths, batch_size=batch_size, num_rx=num_sites)
            if self.require_sionna and getattr(rt_features, 'is_mock', False):
                raise RuntimeError("RT feature extraction returned mock data with require_sionna enabled.")
            if not self._rt_features_logged:
                try:
                    gains = getattr(rt_features, "path_gains", None)
                    if gains is not None:
                        gains_np = gains.numpy() if hasattr(gains, "numpy") else np.asarray(gains)
                        abs_gains = np.abs(gains_np)
                        logger.info(
                            "RT path_gains pre-merge: shape=%s min=%.3e mean=%.3e max=%.3e nonzero=%d",
                            gains_np.shape,
                            float(abs_gains.min()) if abs_gains.size else 0.0,
                            float(abs_gains.mean()) if abs_gains.size else 0.0,
                            float(abs_gains.max()) if abs_gains.size else 0.0,
                            int(np.count_nonzero(abs_gains > 0)),
                        )
                except Exception as stats_e:
                    logger.warning(f"RT path_gains pre-merge stats failed: {stats_e}")
                self._rt_features_logged = True
            
            # 4. Compute Channel Matrix
            channel_matrix = self._compute_channel_matrix(paths, batch_size)
            if channel_matrix is not None:
                try:
                    abs_h = tf.abs(channel_matrix)
                    sample_axes = tuple(range(1, len(abs_h.shape)))
                    sample_max = tf.reduce_max(abs_h, axis=sample_axes)
                    sample_max_np = sample_max.numpy()
                    bad_mask = (~np.isfinite(sample_max_np)) | (sample_max_np <= 0.0)
                    if np.any(bad_mask):
                        bad_indices = np.where(bad_mask)[0].tolist()
                        example_indices = bad_indices[:8]
                        bad_positions = ue_positions[example_indices] if len(ue_positions) > 0 else []
                        self._channel_matrix_fail += int(bad_mask.sum())
                        self._channel_fail_log_count += 1
                        if self._channel_fail_log_count % self._channel_fail_log_every == 0:
                            self._log_failure_diagnostics(
                                "channel_matrix_invalid",
                                ue_positions,
                                site_positions,
                                rt_features=rt_features,
                                indices=example_indices,
                            )
                        err = RuntimeError(
                            f"Channel matrix has {int(bad_mask.sum())}/{len(sample_max_np)} "
                            f"samples with zero/invalid max; example indices={example_indices}, "
                            f"positions={bad_positions}"
                        )
                        setattr(err, "bad_indices", bad_indices)
                        raise err
                except Exception as stats_e:
                    if isinstance(stats_e, RuntimeError):
                        raise
                    logger.warning(f"Per-sample channel matrix check failed: {stats_e}")
            else:
                if self.require_cfr:
                    self._cfr_missing += batch_size
                    self._log_failure_diagnostics(
                        "channel_matrix_missing",
                        ue_positions,
                        site_positions,
                        rt_features=rt_features,
                    )
            
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
            if not self._rt_merge_logged:
                try:
                    gains = getattr(rt_features, "path_gains", None)
                    if gains is not None:
                        gains_np = gains.numpy() if hasattr(gains, "numpy") else np.asarray(gains)
                        abs_gains = np.abs(gains_np)
                        logger.info(
                            "RT path_gains post-merge: shape=%s min=%.3e mean=%.3e max=%.3e nonzero=%d",
                            gains_np.shape,
                            float(abs_gains.min()) if abs_gains.size else 0.0,
                            float(abs_gains.mean()) if abs_gains.size else 0.0,
                            float(abs_gains.max()) if abs_gains.size else 0.0,
                            int(np.count_nonzero(abs_gains > 0)),
                        )
                except Exception as stats_e:
                    logger.warning(f"RT path_gains post-merge stats failed: {stats_e}")
                self._rt_merge_logged = True

            if self.require_cfr:
                cfr = getattr(phy_features, 'cfr_magnitude', None)
                if cfr is None:
                    self._cfr_missing += batch_size
                    self._log_failure_diagnostics(
                        "cfr_missing",
                        ue_positions,
                        site_positions,
                        rt_features=rt_features,
                    )
                    raise RuntimeError("CFR missing after native feature merge with require_cfr enabled.")
                cfr_shape = getattr(cfr, 'shape', None)
                if cfr_shape is not None and len(cfr_shape) < 2:
                    self._cfr_missing += batch_size
                    self._log_failure_diagnostics(
                        "cfr_shape_invalid",
                        ue_positions,
                        site_positions,
                        rt_features=rt_features,
                    )
                    raise RuntimeError(f"Unexpected CFR shape {cfr_shape} with require_cfr enabled.")
            
            return rt_features, phy_features, mac_features
            
        except Exception as e:
            self._sionna_fail += batch_size
            self._channel_fail_log_count += 1
            if self._channel_fail_log_count % self._channel_fail_log_every == 0:
                logger.error(f"Sionna batch simulation failed: {e}")
                logger.error(traceback.format_exc())
            if self.require_sionna or not self.allow_mock_fallback:
                raise
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
        max_paths = getattr(self.config, 'max_num_paths_per_src', None)
        if max_paths is None:
            max_paths = getattr(self.config, 'max_num_paths', None)
        if max_paths is None:
            max_paths = getattr(self.config, 'max_stored_paths', 256)
        max_stored_paths = getattr(self.config, 'max_stored_paths', None)
        if max_stored_paths is not None:
            max_paths = min(max_paths, max_stored_paths)

        paths_result = PathSolver()(
            scene,
            max_depth=max_depth,
            samples_per_src=samples_per_src,
            max_num_paths_per_src=max_paths,
            los=True,
            specular_reflection=True,
            diffuse_reflection=getattr(self.config, 'enable_diffuse_reflection', False),
            refraction=True,
            diffraction=getattr(self.config, 'enable_diffraction', True),
            edge_diffraction=getattr(self.config, 'enable_edge_diffraction', False)
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
            num_subcarriers = getattr(self.config, 'cfr_num_subcarriers', None)
            if num_subcarriers is None:
                num_subcarriers = getattr(self.config, 'num_subcarriers', 1024)
            bw = self.config.bandwidth_hz
            
            # Frequencies: [-BW/2 ... +BW/2]
            freqs = tf.linspace(-bw/2, bw/2, num_subcarriers)
            freqs = tf.cast(freqs, tf.float32)

            # Use Sionna's CIR helper to handle tuple (real, imag) safely.
            # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            # tau: [num_rx, num_tx, num_paths] or [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            a_raw, tau_raw = paths.cir(out_type="tf", num_time_steps=1, normalize_delays=True)
            a = tf.cast(a_raw, tf.complex64)
            tau = tf.cast(tau_raw, tf.float32)
            a_shape = a.shape
            tau_shape = tau.shape

            # cir_to_ofdm_channel expects a batch dimension. Use a batch of 1.
            a = tf.expand_dims(a, axis=0)
            tau = tf.expand_dims(tau, axis=0)

            # Compute Channel Transfer Function
            normalize_cfr = bool(getattr(self.config, 'normalize_cfr', False))
            h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=normalize_cfr)
            if not self._channel_matrix_logged:
                logger.info(f"cir_to_ofdm_channel normalize={normalize_cfr}")

            # h_freq shape: [1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            h_freq = tf.squeeze(h_freq, axis=0)  # drop batch=1
            h_freq = tf.squeeze(h_freq, axis=-2)  # drop time dimension

            # Return shape: [batch(num_rx), 1, rx_ant, num_tx, tx_ant, freq]
            channel_matrix = tf.expand_dims(h_freq, axis=1)

            # Verify shape (batch should match num_rx receivers)
            if channel_matrix.shape[0] != batch_size and channel_matrix.shape[1] == batch_size:
                logger.warning("Transpose flip detected in channel_matrix. Correcting to [Batch, ...]")
                channel_matrix = tf.transpose(channel_matrix, perm=[1, 0, 2, 3, 4, 5])

            try:
                abs_h = tf.abs(channel_matrix)
                max_val = tf.reduce_max(abs_h)
                min_val = tf.reduce_min(abs_h)
                mean_val = tf.reduce_mean(abs_h)
                max_val_np = float(max_val.numpy())
                min_val_np = float(min_val.numpy())
                mean_val_np = float(mean_val.numpy())
                log_stats = (not self._channel_matrix_logged) or (not np.isfinite(max_val_np)) or max_val_np <= 0.0
                if log_stats:
                    logger.info(
                        "Channel matrix |H| stats: min=%.3e mean=%.3e max=%.3e",
                        min_val_np,
                        mean_val_np,
                        max_val_np,
                    )
                if not np.isfinite(max_val_np) or max_val_np <= 0.0:
                    raise RuntimeError(
                        f"Channel matrix is zero/invalid (min={min_val_np:.3e}, "
                        f"mean={mean_val_np:.3e}, max={max_val_np:.3e})."
                    )
            except Exception as stats_e:
                if isinstance(stats_e, RuntimeError):
                    raise
                logger.warning(f"Channel matrix stats check failed: {stats_e}")

            if not self._channel_matrix_logged:
                logger.info(f"Computed channel matrix shape: {channel_matrix.shape}")
                self._channel_matrix_logged = True

            return channel_matrix
            
        except Exception as e:
            if not self._channel_matrix_logged:
                logger.warning(
                    f"Channel matrix computation failed: {e} "
                    f"(a shape={a_shape}, tau shape={tau_shape}, "
                    f"a type={type(a_raw)}, tau type={type(tau_raw)})"
                )
                self._channel_matrix_logged = True
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
        
        # Extract Native PHY Features (including CFR and PMI)
        if channel_matrix is not None:
            phy_native_dict = self.native_extractor.extract_phy(channel_matrix, batch_size)
            
            # Standard metrics
            phy_features.capacity_mbps = phy_native_dict.get('on_se') * (self.config.bandwidth_hz / 1e6)
            phy_features.condition_number = phy_native_dict.get('cond_num')
            
            # PMI - Precoding Matrix Indicator (SVD-based optimal precoder)
            if 'pmi' in phy_native_dict:
                phy_features.pmi = phy_native_dict.get('pmi')
            
            # CFR - Channel Frequency Response (channel estimation from DMRS)
            # This is the key feature representing what the UE estimates from reference signals
            if 'cfr_magnitude' in phy_native_dict:
                phy_features.cfr_magnitude = phy_native_dict.get('cfr_magnitude')
            if 'cfr_phase' in phy_native_dict:
                phy_features.cfr_phase = phy_native_dict.get('cfr_phase')
    
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
            # Generate mock features with all required fields
            num_mock_paths = 64
            rt_m = RTLayerFeatures(
                path_gains=np.random.randn(1, 1, num_mock_paths) * 1e-8,
                path_delays=np.random.rand(1, 1, num_mock_paths) * 1e-6,
                path_aoa_azimuth=np.random.rand(1, 1, num_mock_paths),
                path_aoa_elevation=np.random.rand(1, 1, num_mock_paths),
                path_aod_azimuth=np.random.rand(1, 1, num_mock_paths),
                path_aod_elevation=np.random.rand(1, 1, num_mock_paths),
                path_doppler=np.random.rand(1, 1, num_mock_paths),
                rms_delay_spread=np.random.rand(1, 1) * 1e-7,
                num_paths=np.ones((1, 1), dtype=np.int32) * num_mock_paths
            )
            phy_m = PHYFAPILayerFeatures(
                rsrp=np.random.randn(1, 1, 16) * 10 - 100,
                rsrq=np.random.randn(1, 1, 16) * 5 - 10,
                sinr=np.random.randn(1, 1, 16) * 10,
                cqi=np.random.randint(0, 16, size=(1, 1, 16)),
                ri=np.ones((1, 1, 16), dtype=np.int32),
                pmi=np.zeros((1, 1, 16), dtype=np.int32)
            )
            mac_m = MACRRCLayerFeatures(
                serving_cell_id=np.zeros((1, 1), dtype=np.int32),
                neighbor_cell_ids=np.zeros((1, 1, 8), dtype=np.int32),
                timing_advance=np.random.randint(0, 100, size=(1, 1))
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

    def _log_failure_diagnostics(
        self,
        label: str,
        ue_positions: np.ndarray,
        site_positions: np.ndarray,
        rt_features: Optional[Any] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        if self._diagnostic_logs >= self._diagnostic_max:
            return
        self._diagnostic_logs += 1

        try:
            if indices is not None and len(indices) > 0:
                ue_sel = ue_positions[indices]
            else:
                ue_sel = ue_positions

            if ue_sel is None or len(ue_sel) == 0:
                return

            # Distance to nearest site (2D horizontal).
            if site_positions is not None and len(site_positions) > 0:
                dxy = ue_sel[:, None, :2] - site_positions[None, :, :2]
                dists = np.linalg.norm(dxy, axis=-1)
                min_dist = np.min(dists, axis=1)
                logger.warning(
                    "%s diagnostics: nearest-site distance (m) min=%.1f med=%.1f mean=%.1f max=%.1f",
                    label,
                    float(np.min(min_dist)),
                    float(np.median(min_dist)),
                    float(np.mean(min_dist)),
                    float(np.max(min_dist)),
                )

            # RT path gain / path count summary.
            if rt_features is not None:
                gains = getattr(rt_features, "path_gains", None)
                if gains is not None:
                    gains_np = gains.numpy() if hasattr(gains, "numpy") else np.asarray(gains)
                    gains_abs = np.abs(gains_np)
                    if indices is not None and len(indices) > 0:
                        gains_abs = gains_abs[indices]
                    total_power = np.sum(gains_abs ** 2, axis=(1, 2))
                    power_db = 10.0 * np.log10(total_power + 1e-30)
                    logger.warning(
                        "%s diagnostics: total path power (dB) min=%.1f med=%.1f mean=%.1f max=%.1f",
                        label,
                        float(np.min(power_db)),
                        float(np.median(power_db)),
                        float(np.mean(power_db)),
                        float(np.max(power_db)),
                    )
                    nonzero_paths = np.sum(gains_abs > 0, axis=(1, 2))
                    logger.warning(
                        "%s diagnostics: nonzero paths min=%d med=%d mean=%.1f max=%d",
                        label,
                        int(np.min(nonzero_paths)),
                        int(np.median(nonzero_paths)),
                        float(np.mean(nonzero_paths)),
                        int(np.max(nonzero_paths)),
                    )
                num_paths = getattr(rt_features, "num_paths", None)
                if num_paths is not None:
                    num_paths_np = num_paths.numpy() if hasattr(num_paths, "numpy") else np.asarray(num_paths)
                    if indices is not None and len(indices) > 0:
                        num_paths_np = num_paths_np[indices]
                    zero_mask = (num_paths_np <= 0)
                    frac_zero = float(np.mean(zero_mask)) if num_paths_np.size else 0.0
                    logger.warning(
                        "%s diagnostics: num_paths zero fraction=%.2f",
                        label,
                        frac_zero,
                    )
        except Exception as diag_e:
            logger.warning("Diagnostics logging failed (%s): %s", label, diag_e)
    
    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return {
            'sionna_ok': self._sionna_ok,
            'sionna_fail': self._sionna_fail,
            'channel_matrix_fail': self._channel_matrix_fail,
            'cfr_missing': self._cfr_missing,
        }
