
import pytest
import numpy as np
import tensorflow as tf
from src.data_generation.features import PHYFAPIFeatureExtractor, MACRRCFeatureExtractor, RTLayerFeatures
from src.data_generation.measurement_utils import compute_rsrp
import logging

class TestFeatureDimensionRobustness:
    """
    Validates robustness of feature extractors against dimension permutations/squeezing.
    Focuses on scenarios like Batch >> Sites (e.g., 32 UEs, 1 Site) where unintended broadcasting occurs.
    """

    @pytest.fixture
    def config(self):
        return {
            'batch_size': 32,
            'num_sites': 1,
            'num_rx_ant': 1,
            'num_freq': 16
        }

    def test_mac_extractor_rank2_alignment(self, config):
        """Test MAC extractor handles [Sites, Batch] vs [Batch, Sites] ambiguity."""
        batch_size = config['batch_size']
        num_sites = config['num_sites']
        
        mac_extractor = MACRRCFeatureExtractor()
        ue_pos = np.zeros((batch_size, 3))
        site_pos = np.zeros((num_sites, 3))
        # Ensure distinct IDs
        cell_ids = np.array([101]) if num_sites == 1 else np.arange(101, 101+num_sites)
        
        # Case 1: Correct Shape [Batch, Sites] or [Batch, Sites, 1]
        rsrp_correct = np.zeros((batch_size, num_sites)) 
        # Make batch item 5 correspond to site 0 (trivial for 1 site)
        
        # Test align_dims implicitly via extract()
        # Mock PHY features
        class MockPHY:
            def __init__(self, r, c):
                self.rsrp = r
                self.cqi = c
        
        phy_correct = MockPHY(rsrp_correct, np.zeros_like(rsrp_correct))
        
        
        res = mac_extractor.extract(phy_correct, ue_pos, site_pos, cell_ids)
        assert res.serving_cell_id.shape == (batch_size, 1)
        assert res.serving_cell_id[0, 0] == 101, "Failed to extract with correct shape"

        # Case 2: Swapped/Rank 2 Shape [Sites, Batch] (e.g. [1, 32])
        # This was the crash cause: argmax over axis -1 (32) returned scalar/indices out of site range.
        rsrp_swapped = np.zeros((num_sites, batch_size))
        
        phy_swapped = MockPHY(rsrp_swapped, np.zeros_like(rsrp_swapped))
        
        # This should NOT crash and should return correct IDs
        res_swapped = mac_extractor.extract(phy_swapped, ue_pos, site_pos, cell_ids)
        assert res_swapped.serving_cell_id.shape == (batch_size, 1)
        assert res_swapped.serving_cell_id[5, 0] == 101, "Failed to handle swapped [Sites, Batch]"

    def test_mac_extractor_rank3_alignment(self, config):
        """Test MAC extractor handles [Sites, Batch, 1] vs [Batch, Sites, 1]."""
        batch_size = config['batch_size']
        num_sites = config['num_sites']
        
        mac_extractor = MACRRCFeatureExtractor()
        ue_pos = np.zeros((batch_size, 3))
        site_pos = np.zeros((num_sites, 3))
        cell_ids = np.array([101])
        
        # Case 1: [Batch, Sites, 1] (Correct) -> [32, 1, 1]
        rsrp_correct = np.zeros((batch_size, num_sites, 1))
        phy_correct = type('obj', (object,), {'rsrp': rsrp_correct, 'cqi': rsrp_correct})
        
        res = mac_extractor.extract(phy_correct, ue_pos, site_pos, cell_ids)
        assert res.serving_cell_id.shape == (batch_size, 1)

        # Case 2: [Sites, Batch, 1] (Swapped) -> [1, 32, 1]
        # align_dims should permute to [32, 1, 1]
        rsrp_swapped = np.zeros((num_sites, batch_size, 1))
        phy_swapped = type('obj', (object,), {'rsrp': rsrp_swapped, 'cqi': rsrp_swapped})
        
        res_swapped = mac_extractor.extract(phy_swapped, ue_pos, site_pos, cell_ids)
        assert res_swapped.serving_cell_id.shape == (batch_size, 1)

    def test_phy_fallback_dimensions(self, config):
        """Test PHY extractor fallback logic produces consistent output shapes."""
        batch_size = config['batch_size']
        num_sites = config['num_sites'] # 1
        num_paths = 3
        
        phy_extractor = PHYFAPIFeatureExtractor()
        
        # RT Features [Batch, Sites, Paths]
        rt_features = RTLayerFeatures(
            path_gains=np.ones((batch_size, num_sites, num_paths), dtype=complex),
            path_delays=np.zeros((batch_size, num_sites, num_paths)),
            path_aoa_azimuth=None, path_aoa_elevation=None,
            path_aod_azimuth=None, path_aod_elevation=None,
            path_doppler=None,
            rms_delay_spread=None, rms_angular_spread=None,
            k_factor=None, num_paths=None,
            carrier_frequency_hz=3.5e9, bandwidth_hz=100e6, is_mock=False
        )
        
        # Run Extract (Fallback path)
        phy_features = phy_extractor.extract(rt_features, channel_matrix=None)
        
        # Expect RSRP: [Batch, Sites, 1]
        assert phy_features.rsrp.shape == (batch_size, num_sites, 1)
