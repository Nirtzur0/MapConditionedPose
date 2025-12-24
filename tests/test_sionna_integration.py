"""
Pytest Sionna integration checks.
"""

from pathlib import Path

import pytest
import yaml


def test_sionna_imports():
    """Ensure Sionna + TensorFlow are importable when installed."""
    sionna = pytest.importorskip("sionna")
    pytest.importorskip("tensorflow")

    from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray

    assert hasattr(sionna, "__version__")
    assert load_scene is not None
    assert Transmitter is not None
    assert Receiver is not None
    assert PlanarArray is not None


def test_sionna_config_loading():
    """Validate Sionna configuration file structure."""
    config_path = Path("configs/data_generation/data_generation_sionna.yaml")
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert 'scene_dir' in config
    assert 'carrier_frequency_hz' in config
    assert 'bandwidth_hz' in config
    assert 'num_samples' in config
    assert 'output_dir' in config
