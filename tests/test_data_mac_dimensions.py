
import pytest
import numpy as np
import tensorflow as tf
from src.data_generation.features import MACRRCFeatureExtractor, PHYFAPILayerFeatures

class TestMACDimensionRobustness:
    """
    Methodical tests for MACRRCFeatureExtractor dimension handling.
    Focuses on mismatches between cell_ids, site_positions, and RSRP dimensions.
    """

    @pytest.fixture
    def mac_extractor(self):
        return MACRRCFeatureExtractor(max_neighbors=2)

    @pytest.fixture
    def ue_pos_batch(self):
        # [Batch=10, 3]
        return np.random.randn(10, 3)

    def test_site_positions_mismatch_crash(self, mac_extractor, ue_pos_batch):
        """
        Reproduce crash: cell_ids implies 5 cells, but site_positions has only 1.
        If extract logic assumes indices map 1:1, it will access site_positions[index > 0] and crash.
        """
        batch_size = 10
        num_cells_id = 5
        num_sites_pos = 1 # Mismatch!
        
        cell_ids = np.arange(num_cells_id)
        site_positions = np.zeros((num_sites_pos, 3)) 
        
        # RSRP for 5 cells
        rsrp = np.random.randn(batch_size, num_cells_id)
        
        # Mock PHY features
        phy = type('MockPHY', (), {'rsrp': rsrp, 'cqi': rsrp})()
        
        # This SHOULD fail currently if logical bugs exist
        try:
             res = mac_extractor.extract(phy, ue_pos_batch, site_positions, cell_ids)
             # If it survives, check if it did something reasonable (like clipping or modulo)
             # But strictly, if it tries to look up index 4 in a size-1 array, it must crash.
        except IndexError:
             pytest.fail("MAC extractor crashed with IndexError on site_positions mismatch mismatch as expected (repro).")
        except Exception as e:
             pytest.fail(f"Crashed with unexpected error: {e}")

    def test_tf_tensor_mismatch_crash(self, mac_extractor):
        """Same as above but with TF tensors."""
        if not hasattr(tf, 'constant'): return

        batch_size = 5
        num_cells_id = 10
        num_sites_pos = 2 
        
        cell_ids = np.arange(num_cells_id)
        # Site pos only has 2 entries
        site_positions = np.zeros((num_sites_pos, 3))
        
        ue_pos = tf.random.normal((batch_size, 3))
        
        # RSRP suggests 10 cells available
        rsrp = tf.random.normal((batch_size, num_cells_id))
        
        phy = type('MockPHY', (), {'rsrp': rsrp, 'cqi': rsrp})()
        
        try:
             res = mac_extractor.extract(phy, ue_pos, site_positions, cell_ids)
        except Exception as e:
             # TF might raise InvalidArgumentError for gather indices out of bounds
             pytest.fail(f"TF Path crashed: {e}")

    def test_scalar_cell_id_rank1_case(self, mac_extractor):
        """
        Test extreme edge case: cell_ids is size 1, site_pos size 1.
        But RSRP might be [Batch, 1] or [Batch].
        """
        batch_size = 3
        cell_ids = np.array([55])
        site_positions = np.zeros((1, 3))
        ue_pos = np.zeros((batch_size, 3))
        
        # RSRP is [Batch] (rank 1)
        rsrp = np.ones((batch_size,)) * -80
        
        phy = type('MockPHY', (), {'rsrp': rsrp, 'cqi': rsrp})()
        
        # Should handle rank 1 rsrp by treating it as 1 cell?
        res = mac_extractor.extract(phy, ue_pos, site_positions, cell_ids)
        
        assert res.serving_cell_id.shape[0] == batch_size
        assert np.all(res.serving_cell_id == 55)

    def test_empty_inputs_safety(self, mac_extractor):
        """Test with 0 batch size."""
        cell_ids = np.array([1, 2])
        site_positions = np.zeros((2, 3))
        ue_pos = np.zeros((0, 3))
        rsrp = np.zeros((0, 2))
        
        phy = type('MockPHY', (), {'rsrp': rsrp, 'cqi': rsrp})()
        
        res = mac_extractor.extract(phy, ue_pos, site_positions, cell_ids)
        assert len(res.serving_cell_id) == 0

