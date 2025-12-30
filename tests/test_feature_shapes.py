
import pytest
import numpy as np
import logging
from src.data_generation.features.rt_extractor import RTFeatureExtractor
from src.data_generation.config import DataGenerationConfig
from src.data_generation.features.data_structures import RTLayerFeatures

# Mock "Sionna" Paths object

def test_rt_extractor_fixed_output_shape():
    """Verify RTFeatureExtractor enforces max_stored_paths/sites."""
    
    max_sites = 4
    max_paths = 10
    
    extractor = RTFeatureExtractor(
        max_stored_paths=max_paths,
        max_stored_sites=max_sites
    )
    
    # Mock "Sionna" Paths object
    class MockPaths:
        def __init__(self, s, t, p):
            self.a = np.ones((s, t, p, 1, 1, 1, 1), dtype=np.complex64)
            self.tau = np.zeros((s, t, p))
            self.phi_r = np.zeros((s, t, p))
            self.theta_r = np.zeros((s, t, p))
            self.phi_t = np.zeros((s, t, p))
            self.theta_t = np.zeros((s, t, p))
            self.doppler = np.zeros((s, t, p))
            

    # Patch SIONNA_AVAILABLE to True to force generic extraction
    # and patch get_ops to return NumpyOps (which now supports pad)
    import src.data_generation.features.rt_extractor as rt_mod
    from src.data_generation.features.tensor_ops import NumpyOps
    
    rt_mod.SIONNA_AVAILABLE = True
    
    # Store original get_ops
    orig_get_ops = rt_mod.get_ops
    rt_mod.get_ops = lambda x: NumpyOps()
    
    try:
        # Case 1: Smaller than max (Padding required)
        # Batch=2 (Target), Sites=2 (Source), Paths=5
        # Expected: (2, 4, 10)
        batch_size = 2
        actual_sites = 2
        actual_paths = 5
        
        # Paths(Sources=2, Targets=2, Paths=5)
        # Note: Layout in generic extraction is standardized.
        # rt_extractor expects [Sources, Targets, Paths] initially from Sionna
        # And it standardizes to [Batch(Targets), Rx(Sites/Sources), Paths]
        # So input should be [Sources=2, Targets=2, Paths=5]
        paths_small = MockPaths(actual_sites, batch_size, actual_paths)
        
        features = extractor.extract(paths_small, batch_size=batch_size, num_rx=actual_sites)
        
        # Check dimensions
        # RTLayerFeatures stores as (Batch, Sites, Paths)
        print(f"Shape: {features.path_gains.shape}")
        assert features.path_gains.shape == (batch_size, max_sites, max_paths)
        assert features.path_delays.shape == (batch_size, max_sites, max_paths)
        
        # Verify padding (indices >= actual should be 0)
        # Sites: indices 2,3 should be 0
        assert np.all(features.path_gains[:, 2:, :] == 0)
        # Paths: indices 5-9 should be 0
        assert np.all(features.path_gains[:, :2, 5:] == 0)

        # Case 2: Larger than max (Truncation required)
        # Sites=6, Paths=15
        actual_sites_large = 6
        actual_paths_large = 15
        paths_large = MockPaths(actual_sites_large, batch_size, actual_paths_large)
        
        features_large = extractor.extract(paths_large, batch_size=batch_size, num_rx=actual_sites_large)
        
        assert features_large.path_gains.shape == (batch_size, max_sites, max_paths)
        
    finally:
        # Restore
        rt_mod.get_ops = orig_get_ops

    
    # Verify we kept the first N
    # We set values to ones, so check if not zero?
    # Actually, let's set values to indices to verify correctness?
    # For now, shape check is sufficient for this task.

if __name__ == "__main__":
    test_rt_extractor_fixed_output_shape()
    print("Test passed!")
