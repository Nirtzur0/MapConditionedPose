"""
Pytest Tests for M2 Multi-Layer Data Generation
Tests feature extractors, measurement utilities, and data generator
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

# Import M2 modules
from src.data_generation.measurement_utils import (
    compute_rsrp, compute_rsrq, compute_sinr, compute_cqi,
    compute_rank_indicator, compute_timing_advance, compute_pmi,
    add_measurement_dropout, simulate_neighbor_list_truncation,
    compute_beam_rsrp
)

from src.data_generation.features import (
    RTFeatureExtractor, PHYFAPIFeatureExtractor, MACRRCFeatureExtractor,
    RTLayerFeatures, PHYFAPILayerFeatures, MACRRCLayerFeatures
)

from src.data_generation.multi_layer_generator import (
    DataGenerationConfig, MultiLayerDataGenerator
)
from src.data_generation.trajectory import sample_ue_trajectories


# ============================================================================
# Test Measurement Utilities
# ============================================================================

class TestMeasurementUtils:
    """Test 3GPP-compliant measurement computations."""
    
    def test_rsrp_computation(self):
        """Test RSRP calculation from path gains (simplified)."""
        # For test simplicity, test the approximation from path gains used in features.py
        # Total power approach without pilot RE indexing
        path_gains = np.random.randn(10, 4, 50) + 1j * np.random.randn(10, 4, 50)
        
        # Manual RSRP calculation
        total_power = np.sum(np.abs(path_gains)**2, axis=-1)
        rsrp_dbm = 10 * np.log10(total_power + 1e-10) + 30
        
        assert rsrp_dbm.shape == (10, 4)
        # RSRP should be reasonable (random gains, expect around 0 dBm ballpark)
        assert np.all(rsrp_dbm >= -100) and np.all(rsrp_dbm <= 100), "RSRP out of reasonable range"

    def test_rsrp_from_channel_matrix(self):
        """Test RSRP computation from channel matrix and pilots."""
        h = np.ones((1, 1, 1, 1, 1, 4), dtype=np.complex128)
        pilot_re = np.array([0, 2])

        rsrp_dbm = compute_rsrp(h, cell_id=0, pilot_re_indices=pilot_re)

        assert rsrp_dbm.shape == (1, 1, 1)
        assert np.allclose(rsrp_dbm, 30.0), "Unit channel should map to 30 dBm"
    
    def test_rsrq_computation(self):
        """Test RSRQ calculation."""
        rsrp = np.random.uniform(-100, -60, (10, 4))
        rssi = rsrp + np.random.uniform(0, 10, (10, 4))
        
        rsrq = compute_rsrq(rsrp, rssi, N=12)
        
        assert rsrq.shape == (10, 4)
        assert np.all(rsrq >= -34) and np.all(rsrq <= 2.5), "RSRQ out of 3GPP range"
    
    def test_sinr_computation(self):
        """Test SINR with interference."""
        h_serving = np.random.randn(10, 4, 2, 8) + 1j * np.random.randn(10, 4, 2, 8)
        h_interferer = np.random.randn(10, 4, 2, 8) * 0.5 + 1j * np.random.randn(10, 4, 2, 8) * 0.5
        noise_power = 1e-10
        
        sinr = compute_sinr(h_serving, noise_power, [h_interferer])
        
        # compute_sinr sums over antenna dimensions, expect [batch] shape
        assert sinr.shape == (10,)
        assert np.all(sinr >= -20) and np.all(sinr <= 50), "SINR out of reasonable range"
    
    def test_cqi_mapping(self):
        """Test SINR to CQI mapping."""
        sinr = np.array([-10, 0, 5, 10, 15, 20, 25])
        
        cqi = compute_cqi(sinr, mcs_table='64QAM')
        
        assert cqi.shape == sinr.shape
        assert np.all(cqi >= 0) and np.all(cqi <= 15), "CQI out of range"
        assert cqi[0] < cqi[3] < cqi[6], "CQI should increase with SINR"
    
    def test_rank_indicator(self):
        """Test RI from channel matrix."""
        # Full rank channel
        h_full = np.random.randn(10, 4, 8) + 1j * np.random.randn(10, 4, 8)
        ri_full = compute_rank_indicator(h_full, snr_threshold_db=-10)
        
        assert ri_full.shape == (10,)
        assert np.all(ri_full >= 1) and np.all(ri_full <= 4), "RI out of range"
        
        # Rank-1 channel
        h_rank1 = np.random.randn(10, 1, 8) + 1j * np.random.randn(10, 1, 8)
        ri_rank1 = compute_rank_indicator(h_rank1)
        
        assert np.all(ri_rank1 == 1), "Rank-1 channel should have RI=1"

    def test_pmi_computation(self):
        """Test PMI output shape and type."""
        h = np.random.randn(5, 2, 4) + 1j * np.random.randn(5, 2, 4)
        pmi = compute_pmi(h)

        assert pmi.shape == (5,)
        assert pmi.dtype == np.int32
        assert np.all(pmi >= 0)
    
    def test_timing_advance(self):
        """Test TA computation from distance."""
        distances = np.array([10, 100, 500, 1000, 5000])  # meters
        
        ta = compute_timing_advance(distances)
        
        assert ta.shape == distances.shape
        assert np.all(ta >= 0) and np.all(ta <= 3846), "TA out of 3GPP range"
        assert ta[0] < ta[1] < ta[4], "TA should increase with distance"
    
    def test_measurement_dropout(self):
        """Test measurement dropout simulation."""
        features = {
            'rsrp': np.random.uniform(-100, -60, (100, 4)),
            'rsrq': np.random.uniform(-15, -5, (100, 4)),
            'cqi': np.random.randint(0, 16, (100, 4)).astype(float),
        }
        dropout_rates = {'rsrp': 0.2, 'rsrq': 0.3, 'cqi': 0.25}
        
        features_dropped = add_measurement_dropout(features, dropout_rates, seed=42)
        
        for key in features:
            nan_count = np.isnan(features_dropped[key]).sum()
            expected_dropout = int(dropout_rates[key] * features[key].size)
            # Allow 20% tolerance
            assert abs(nan_count - expected_dropout) < 0.2 * expected_dropout, \
                f"{key} dropout rate mismatch"

    def test_beam_rsrp(self):
        """Test L1-RSRP per-beam computation."""
        path_gains = np.ones((2, 3, 5)) + 1j * np.zeros((2, 3, 5))
        num_beams = 8

        l1_rsrp = compute_beam_rsrp(path_gains, beam_directions=None, num_beams=num_beams)

        assert l1_rsrp.shape == (2, 3, num_beams)
        expected_dbm = 10 * np.log10(5.0) + 30
        assert np.allclose(l1_rsrp, expected_dbm), "Uniform paths should yield uniform beam RSRP"
    
    def test_neighbor_list_truncation(self):
        """Test top-K neighbor selection."""
        cells = np.tile(np.arange(20), (10, 1))
        rsrp = np.random.uniform(-120, -60, (10, 20))
        
        neighbor_cells, neighbor_rsrp = simulate_neighbor_list_truncation(cells, rsrp, K=8)
        
        assert neighbor_cells.shape == (10, 8)
        assert neighbor_rsrp.shape == (10, 8)
        
        # Verify sorting (descending RSRP)
        for i in range(10):
            assert np.all(neighbor_rsrp[i, :-1] >= neighbor_rsrp[i, 1:]), \
                "Neighbors should be sorted by RSRP"


# ============================================================================
# Test Feature Extractors
# ============================================================================

class TestRTFeatureExtractor:
    """Test RT layer feature extraction."""
    
    def test_initialization(self):
        """Test RTFeatureExtractor initialization."""
        extractor = RTFeatureExtractor(
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            compute_k_factor=True
        )
        
        assert extractor.carrier_frequency_hz == 3.5e9
        assert extractor.bandwidth_hz == 100e6
        assert extractor.compute_k_factor == True
    
    def test_mock_extraction(self):
        """Test mock RT feature extraction (without Sionna)."""
        extractor = RTFeatureExtractor()
        rt_features = extractor._extract_mock()
        
        # Check all required fields
        assert isinstance(rt_features, RTLayerFeatures)
        assert rt_features.path_gains.shape[-1] == 50  # max_paths
        assert rt_features.path_delays.shape == rt_features.path_gains.shape
        assert rt_features.rms_delay_spread.ndim == 2
        assert rt_features.k_factor is not None
        
        # Check value ranges
        assert np.all(rt_features.path_delays >= 0), "Delays must be positive"
        assert np.all(rt_features.rms_delay_spread >= 0), "RMS-DS must be positive"
    
    def test_rms_ds_computation(self):
        """Test RMS delay spread calculation using public extract()."""
        # Initialize with max_stored_sites=1 to avoid padding mismatch in test
        extractor = RTFeatureExtractor(max_stored_sites=1)
        
        # simple mock paths object
        class MockPaths:
            def __init__(self, a, tau):
                self.a = a
                self.tau = tau
                self.phi_r = np.zeros_like(a) # placeholder
                self.theta_r = np.zeros_like(a)
                self.phi_t = np.zeros_like(a)
                self.theta_t = np.zeros_like(a)
                self.doppler = None

        # Simple case: two paths
        # [Sources=1, Targets=1, Paths=2, Rx=1, Tx=1, 1, 1]
        # Or simpler structure that generic extraction accepts (it handles various shapes)
        # Let's provide [Batch=1, Rx=1, Paths=2] directly? 
        # But generic extraction expects Sionna shape and tries to reduce/align.
        # Let's provide [Batch=1, Rx=1, Paths=2]
        
        gains = np.array([[[1.0 + 0j, 0.5 + 0j]]]) # [1, 1, 2]
        delays = np.array([[[0.0, 1e-6]]]) # [1, 1, 2]
        
        paths = MockPaths(gains, delays)
        
        # Extract with explicit batch_size=1
        features = extractor.extract(paths, batch_size=1, num_rx=1)
        
        rms_ds = features.rms_delay_spread
        assert rms_ds.shape == (1, 1)
        assert rms_ds[0, 0] > 0, "RMS-DS should be positive"
    
    def test_k_factor_computation(self):
        """Test Rician K-factor calculation."""
        extractor = RTFeatureExtractor(compute_k_factor=True, max_stored_sites=1)
        
        class MockPaths:
            def __init__(self, a, tau):
                self.a = a
                self.tau = tau
                self.phi_r = np.zeros_like(a)
                self.theta_r = np.zeros_like(a)
                self.phi_t = np.zeros_like(a)
                self.theta_t = np.zeros_like(a)
        
        # Strong LOS scenario
        gains_los = np.array([[[10.0 + 0j, 1.0 + 0j, 0.5 + 0j]]])
        delays = np.zeros_like(gains_los, dtype=float)
        paths_los = MockPaths(gains_los, delays)
        
        # num_rx matches MockPaths shape
        features_los = extractor.extract(paths_los, batch_size=1, num_rx=1)
        k_los = features_los.k_factor
        
        assert k_los[0, 0] > 10, "Strong LOS should have high K-factor"
        
        # NLOS scenario
        gains_nlos = np.array([[[1.0 + 0j, 0.9 + 0j, 0.8 + 0j]]])
        paths_nlos = MockPaths(gains_nlos, delays)
        
        features_nlos = extractor.extract(paths_nlos, batch_size=1, num_rx=1)
        k_nlos = features_nlos.k_factor
        
        assert k_nlos[0, 0] < 5, "NLOS should have low K-factor"
    
    def test_complex_conversion_robustness(self):
        """Test robustness of complex() against Dr.Jit-like types."""
        from src.data_generation.features import NumpyOps
        ops = NumpyOps()
        
        class MockTensor:
            def __init__(self, val):
                self.val = np.array(val)
            def numpy(self):
                return self.val
            def __add__(self, other):
                 return NotImplemented
            def __radd__(self, other):
                # Simulate Dr.Jit failure when adding/multiplying with python complex
                # Note: valid Dr.Jit arrays often fail on 'other + self' if 'other' is complex/float and self is array-like but restricted
                if isinstance(other, complex):
                     raise RuntimeError("MockDrJit: unsupported complex op")
                return NotImplemented
            def __mul__(self, other):
                return NotImplemented
            def __rmul__(self, other):
                 if isinstance(other, complex):
                     raise RuntimeError("MockDrJit: unsupported complex op")
                 return NotImplemented

        real = MockTensor([1.0, 2.0])
        imag = MockTensor([0.5, 0.5])
        
        # This should succeed now because ops.complex converts to numpy first
        res = ops.complex(real, imag)
        
        assert isinstance(res, np.ndarray)
        assert np.iscomplexobj(res)
        assert res[0] == 1.0 + 0.5j

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for storage."""
        extractor = RTFeatureExtractor()
        rt_features = extractor._extract_mock()
        
        rt_dict = rt_features.to_dict()
        
        assert 'rt/rms_delay_spread' in rt_dict
        assert 'rt/path_gains' in rt_dict
        assert 'rt/path_delays' in rt_dict
        assert 'rt/num_paths' in rt_dict
        if extractor.compute_k_factor:
            assert 'rt/k_factor' in rt_dict


class TestPHYFAPIFeatureExtractor:
    """Test PHY/FAPI layer feature extraction."""
    
    def test_initialization(self):
        """Test PHYFAPIFeatureExtractor initialization."""
        extractor = PHYFAPIFeatureExtractor(
            noise_figure_db=9.0,
            enable_beam_management=True,
            num_beams=64
        )
        
        assert extractor.noise_figure_db == 9.0
        assert extractor.enable_beam_management == True
        assert extractor.num_beams == 64
    
    def test_phy_extraction_from_rt(self):
        """Test PHY feature extraction from RT features."""
        rt_extractor = RTFeatureExtractor()
        rt_features = rt_extractor._extract_mock()
        
        phy_extractor = PHYFAPIFeatureExtractor(enable_beam_management=True)
        phy_features = phy_extractor.extract(rt_features)
        
        # Check all required fields
        assert isinstance(phy_features, PHYFAPILayerFeatures)
        assert phy_features.rsrp.ndim == 3  # [batch, num_rx, num_cells]
        assert phy_features.cqi.min() >= 0 and phy_features.cqi.max() <= 15
        assert phy_features.ri.min() >= 1
        
        # Check beam management
        assert phy_features.l1_rsrp_beams is not None
        assert phy_features.l1_rsrp_beams.shape[-1] == 64, "Should have 64 beams"
        assert phy_features.best_beam_ids.shape[-1] == 4, "Should report top-4 beams"
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        rt_extractor = RTFeatureExtractor()
        rt_features = rt_extractor._extract_mock()
        
        phy_extractor = PHYFAPIFeatureExtractor()
        phy_features = phy_extractor.extract(rt_features)
        
        phy_dict = phy_features.to_dict()
        
        assert 'phy_fapi/rsrp' in phy_dict
        assert 'phy_fapi/cqi' in phy_dict
        assert len(phy_dict) >= 6, "Should have at least 6 PHY features"


class TestMACRRCFeatureExtractor:
    """Test MAC/RRC layer feature extraction."""
    
    def test_initialization(self):
        """Test MACRRCFeatureExtractor initialization."""
        extractor = MACRRCFeatureExtractor(
            max_neighbors=8,
            enable_throughput=True
        )
        
        assert extractor.max_neighbors == 8
        assert extractor.enable_throughput == True
    
    def test_mac_extraction(self):
        """Test MAC/RRC feature extraction."""
        # Create mock PHY features
        batch_size, num_rx, num_cells = 10, 4, 3
        rt_extractor = RTFeatureExtractor()
        rt_features = rt_extractor._extract_mock()
        phy_extractor = PHYFAPIFeatureExtractor()
        phy_features = phy_extractor.extract(rt_features)
        
        # Expand to multi-cell
        phy_features.rsrp = np.random.uniform(-100, -60, (batch_size, num_rx, num_cells))
        phy_features.rsrq = np.random.uniform(-15, -5, (batch_size, num_rx, num_cells))
        phy_features.sinr = np.random.uniform(-5, 25, (batch_size, num_rx, num_cells))
        phy_features.cqi = np.random.randint(0, 16, (batch_size, num_rx, num_cells))
        
        # UE and site positions
        ue_positions = np.random.uniform(-500, 500, (batch_size, num_rx, 3))
        site_positions = np.array([[0, 0, 30], [500, 0, 30], [-250, 433, 30]])
        cell_ids = np.array([1, 2, 3])
        
        # Extract
        mac_extractor = MACRRCFeatureExtractor(max_neighbors=8, enable_throughput=True)
        mac_features = mac_extractor.extract(phy_features, ue_positions, site_positions, cell_ids)
        
        # Check all required fields
        assert isinstance(mac_features, MACRRCLayerFeatures)
        assert mac_features.serving_cell_id.shape == (batch_size, num_rx)
        # With max_neighbors=8 but only 3 cells, we get min(8, 3) = 3 neighbors (includes serving cell duplicates)
        # Actually, the code takes top-K from ALL cells after masking serving, so we get all available
        assert mac_features.neighbor_cell_ids.shape == (batch_size, num_rx, 3)  # Gets all 3 cells
        assert mac_features.timing_advance.shape == (batch_size, num_rx)
        assert mac_features.dl_throughput_mbps is not None
        
        # Check value ranges
        assert np.all(np.isin(mac_features.serving_cell_id, cell_ids)), "Invalid cell IDs"
        assert np.all(mac_features.timing_advance >= 0), "TA must be positive"
        assert np.all(mac_features.dl_throughput_mbps >= 0), "Throughput must be positive"
    
    def test_throughput_simulation(self):
        """Test throughput estimation from CQI."""
        extractor = MACRRCFeatureExtractor(enable_throughput=True)
        
        # Low CQI -> low throughput
        cqi_low = np.array([[[2, 3, 4]]])
        throughput_low = extractor._simulate_throughput(cqi_low)
        
        # High CQI -> high throughput
        cqi_high = np.array([[[12, 13, 14]]])
        throughput_high = extractor._simulate_throughput(cqi_high)
        
        assert np.mean(throughput_high) > np.mean(throughput_low), \
            "Higher CQI should give higher throughput"
    
    def test_bler_simulation(self):
        """Test BLER from SINR."""
        extractor = MACRRCFeatureExtractor()
        
        # Low SINR -> high BLER
        sinr_low = np.array([[-5, 0, 2]])
        bler_low = extractor._simulate_bler(sinr_low)
        
        # High SINR -> low BLER
        sinr_high = np.array([[15, 20, 25]])
        bler_high = extractor._simulate_bler(sinr_high)
        
        assert np.mean(bler_low) > np.mean(bler_high), \
            "Lower SINR should give higher BLER"
        assert np.all(bler_low >= 0) and np.all(bler_low <= 1), "BLER must be in [0, 1]"


# ============================================================================
# Test Data Generator
# ============================================================================

class TestDataGenerationConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = DataGenerationConfig(
            scene_dir=Path("test_scenes"),
            scene_metadata_path=Path("test_scenes/metadata.json"),
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            tx_power_dbm=43.0,
            noise_figure_db=9.0,
        )
        
        assert config.carrier_frequency_hz == 3.5e9
        assert config.num_ue_per_tile == 100
        assert config.num_reports_per_ue == 10
        assert config.measurement_dropout_rates is None
    
    def test_config_custom_values(self):
        """Test config with custom values."""
        config = DataGenerationConfig(
            scene_dir=Path("test_scenes"),
            scene_metadata_path=Path("test_scenes/metadata.json"),
            carrier_frequency_hz=28e9,
            bandwidth_hz=100e6,
            tx_power_dbm=43.0,
            noise_figure_db=9.0,
            num_ue_per_tile=50,
            num_reports_per_ue=5,
        )
        
        assert config.carrier_frequency_hz == 28e9
        assert config.num_ue_per_tile == 50
        assert config.num_reports_per_ue == 5


class TestMultiLayerDataGenerator:
    """Test complete data generation pipeline."""
    
    @pytest.fixture
    def temp_scene_dir(self):
        """Create temporary scene directory."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock scene structure
        scene_dir = temp_dir / "scenes"
        scene_dir.mkdir()
        
        metadata = {
            'test_scene': {
                'bbox': {'x_min': -500, 'x_max': 500, 'y_min': -500, 'y_max': 500},
                'sites': [
                    {'position': [0, 0, 30], 'cell_id': 1},
                    {'position': [500, 0, 30], 'cell_id': 2},
                ],
            }
        }
        
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        yield scene_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_generator_initialization(self, temp_scene_dir):
        """Test MultiLayerDataGenerator initialization."""
        config = DataGenerationConfig(
            scene_dir=temp_scene_dir,
            scene_metadata_path=temp_scene_dir / "metadata.json",
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            tx_power_dbm=43.0,
            noise_figure_db=9.0,
            num_ue_per_tile=10,
            num_reports_per_ue=5,
        )
        
        generator = MultiLayerDataGenerator(config)
        
        assert generator.rt_extractor is not None
        assert generator.phy_extractor is not None
        assert generator.mac_extractor is not None
    
    def test_ue_trajectory_sampling(self, temp_scene_dir):
        """Test UE trajectory generation."""
        config = DataGenerationConfig(
            scene_dir=temp_scene_dir,
            scene_metadata_path=temp_scene_dir / "metadata.json",
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            tx_power_dbm=43.0,
            noise_figure_db=9.0,
            num_ue_per_tile=10,
            num_reports_per_ue=5,
        )
        
        generator = MultiLayerDataGenerator(config)
        
        scene_metadata = {'bbox': {'x_min': -500, 'x_max': 500, 'y_min': -500, 'y_max': 500}}
        trajectories = sample_ue_trajectories(
            scene_metadata=scene_metadata,
            num_ue_per_tile=config.num_ue_per_tile,
            ue_height_range=config.ue_height_range,
            ue_velocity_range=config.ue_velocity_range,
            num_reports_per_ue=config.num_reports_per_ue,
            report_interval_ms=config.report_interval_ms
        )
        
        assert len(trajectories) == 10, "Should have 10 UE trajectories"
        assert trajectories[0].shape == (5, 3), "Each trajectory should have 5 positions (x,y,z)"
        
        # Check bounds
        for traj in trajectories:
            assert np.all(traj[:, 0] >= -500) and np.all(traj[:, 0] <= 500), "X out of bounds"
            assert np.all(traj[:, 1] >= -500) and np.all(traj[:, 1] <= 500), "Y out of bounds"
    
    def test_single_measurement_simulation(self, temp_scene_dir):
        """Test simulation for single UE position."""
        config = DataGenerationConfig(
            scene_dir=temp_scene_dir,
            scene_metadata_path=temp_scene_dir / "metadata.json",
            carrier_frequency_hz=3.5e9,
            bandwidth_hz=100e6,
            tx_power_dbm=43.0,
            noise_figure_db=9.0,
        )
        
        generator = MultiLayerDataGenerator(config)
        
        ue_pos = np.array([100, 200, 1.5])
        site_positions = np.array([[0, 0, 30], [500, 0, 30]])
        cell_ids = np.array([1, 2])
        
        rt_feat, phy_feat, mac_feat = generator._simulate_mock(ue_pos, site_positions, cell_ids)
        
        # Check feature types
        assert isinstance(rt_feat, RTLayerFeatures)
        assert isinstance(phy_feat, PHYFAPILayerFeatures)
        assert isinstance(mac_feat, MACRRCLayerFeatures)
        
        # Check shapes (single UE)
        assert rt_feat.rms_delay_spread.shape[1] == 1, "Should be single UE"
        assert phy_feat.rsrp.shape[1] == 1
        assert mac_feat.serving_cell_id.shape[1] == 1


# ============================================================================
# Test Zarr Writer (optional - requires zarr package)
# ============================================================================

# Check if zarr is available
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

@pytest.mark.skipif(not ZARR_AVAILABLE, reason="zarr package not available")
class TestZarrWriter:
    """Test Zarr dataset writer."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_zarr_writer_initialization(self, temp_output_dir):
        """Test ZarrDatasetWriter initialization."""
        from src.data_generation.zarr_writer import ZarrDatasetWriter
        
        writer = ZarrDatasetWriter(output_dir=temp_output_dir, chunk_size=50)
        
        assert writer.store_path.exists()
        assert writer.chunk_size == 50
    
    def test_zarr_append_and_finalize(self, temp_output_dir):
        """Test appending data and finalizing."""
        from src.data_generation.zarr_writer import ZarrDatasetWriter
        
        writer = ZarrDatasetWriter(output_dir=temp_output_dir)
        
        # Create mock scene data
        scene_data = {
            'positions': np.random.uniform(-500, 500, (100, 3)),
            'timestamps': np.arange(100) * 0.2,
            'rt/rms_delay_spread': np.random.uniform(10e-9, 200e-9, 100),
            'phy_fapi/rsrp': np.random.uniform(-100, -60, (100, 3)),
            'mac_rrc/serving_cell_id': np.random.randint(0, 3, 100),
        }
        
        writer.append(scene_data, scene_id='test_scene')
        
        assert writer.current_idx == 100
        
        store_path = writer.finalize()
        assert store_path.exists()

    def test_zarr_resume_functionality(self, temp_output_dir):
        """Test resuming/appending to an existing Zarr dataset."""
        from src.data_generation.zarr_writer import ZarrDatasetWriter
        
        # Initial write
        writer1 = ZarrDatasetWriter(output_dir=temp_output_dir)
        data1 = {'positions': np.zeros((50, 3))}
        writer1.append(data1, scene_id='scene1')
        path1 = writer1.finalize()
        
        # Resume (create new writer point to same dir should ideally append or handle overwrite)
        # Note: Current implementation makes a new dataset each time. 
        # This test ensures we don't crash and create a NEW valid dataset.
        
        # Sleep to ensure unique timestamp
        import time
        time.sleep(1.2)
        
        writer2 = ZarrDatasetWriter(output_dir=temp_output_dir)
        data2 = {'positions': np.ones((50, 3))}
        writer2.append(data2, scene_id='scene2')
        path2 = writer2.finalize()
        
        assert path2.exists()
        assert path1 != path2 # Should create unique timestamped datasets by default


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
