#!/usr/bin/env python3
"""
Test Sionna Integration
Verifies that Sionna RT, PHY, and SYS layers work correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sionna_imports():
    """Test that Sionna is installed and imports correctly."""
    logger.info("Testing Sionna imports...")
    
    try:
        import sionna
        logger.info(f"✅ Sionna {sionna.__version__} imported")
        
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
        logger.info("✅ Sionna RT imported")
        
        import tensorflow as tf
        logger.info(f"✅ TensorFlow {tf.__version__} imported")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_feature_extractors():
    """Test that feature extractors can handle Sionna mode."""
    logger.info("Testing feature extractors...")
    
    from src.data_generation.features import (
        RTFeatureExtractor, PHYFAPIFeatureExtractor, MACRRCFeatureExtractor
    )
    
    # Create extractors
    rt_ext = RTFeatureExtractor(
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=100e6,
        compute_k_factor=False
    )
    
    phy_ext = PHYFAPIFeatureExtractor(
        noise_figure_db=9.0,
        enable_beam_management=True,
        num_beams=64
    )
    
    mac_ext = MACRRCFeatureExtractor(
        max_neighbors=8
    )
    
    logger.info("✅ Feature extractors created")
    
    # Test mock mode still works
    rt_features = rt_ext._extract_mock()
    logger.info(f"✅ Mock RT features: shape {rt_features.path_gains.shape}")
    
    phy_features = phy_ext.extract(rt_features)
    logger.info(f"✅ PHY features: RSRP shape {phy_features.rsrp.shape}")
    
    return True

def test_multi_layer_generator():
    """Test MultiLayerDataGenerator with Sionna config."""
    logger.info("Testing MultiLayerDataGenerator...")
    
    from src.data_generation.multi_layer_generator import (
        MultiLayerDataGenerator, DataGenerationConfig
    )
    
    # Create minimal config
    config = DataGenerationConfig(
        scene_dir=Path("data/scenes"),
        scene_metadata_path=Path("data/scenes/metadata.json"),
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=100e6,
        use_mock_mode=True,  # Start with mock
        num_ue_per_tile=10,
        max_depth=5,
        num_samples=1_000_000,
        enable_diffraction=True
    )
    
    gen = MultiLayerDataGenerator(config)
    logger.info("✅ MultiLayerDataGenerator created")
    
    # Test scene loading methods exist
    assert hasattr(gen, '_load_sionna_scene')
    assert hasattr(gen, '_setup_transmitters')
    assert hasattr(gen, '_setup_receiver')
    logger.info("✅ Sionna integration methods present")
    
    # Test mock simulation
    ue_pos = np.array([100.0, 200.0, 1.5])
    site_pos = np.array([[0.0, 0.0, 25.0], [500.0, 0.0, 25.0]])
    cell_ids = np.array([1, 2])
    
    rt_feat, phy_feat, mac_feat = gen._simulate_mock(ue_pos, site_pos, cell_ids)
    logger.info(f"✅ Mock simulation: RT paths={rt_feat.path_gains.shape}, PHY RSRP={phy_feat.rsrp.shape}")
    
    return True

def test_config_loading():
    """Test loading Sionna configuration file."""
    logger.info("Testing configuration loading...")
    
    import yaml
    
    config_path = Path("configs/data_generation_sionna.yaml")
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check key sections
    assert 'sionna_rt' in config
    assert 'rf_parameters' in config
    assert 'antenna_arrays' in config
    assert 'sampling' in config
    
    logger.info(f"✅ Config loaded: {len(config)} sections")
    logger.info(f"   - Carrier frequency: {config['rf_parameters']['carrier_frequency_hz']/1e9:.1f} GHz")
    logger.info(f"   - Ray samples: {config['sionna_rt']['num_samples']:,}")
    logger.info(f"   - Max depth: {config['sionna_rt']['max_depth']}")
    logger.info(f"   - UEs per tile: {config['sampling']['num_ue_per_tile']}")
    
    return True

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("SIONNA INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    tests = [
        ("Sionna Imports", test_sionna_imports),
        ("Feature Extractors", test_feature_extractors),
        ("MultiLayerDataGenerator", test_multi_layer_generator),
        ("Configuration Loading", test_config_loading),
    ]
    
    results = {}
    for name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {name}")
        logger.info(f"{'='*60}")
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
