"""
Tests for Sionna-specific capability handling.

These tests avoid a hard Sionna dependency by using synthetic Path-like objects.
"""

import numpy as np
import pytest

from src.data_generation.features import RTFeatureExtractor
import src.data_generation.features as features_module
import src.physics_loss.radio_map_generator as radio_map_generator


class FakePaths:
    """Minimal Sionna Paths-like object for extractor tests."""

    def __init__(self, a, tau, phi_r, theta_r, phi_t, theta_t, doppler=None):
        self.a = a
        self.tau = tau
        self.phi_r = phi_r
        self.theta_r = theta_r
        self.phi_t = phi_t
        self.theta_t = theta_t
        if doppler is not None:
            self.doppler = doppler


def _make_paths(include_doppler=True):
    num_rx = 1
    num_rx_ant = 2
    num_tx = 2
    num_tx_ant = 2
    num_paths = 3
    num_time = 4

    path_values = np.arange(1, num_paths + 1, dtype=np.float64)
    real = (
        np.ones((num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time))
        * path_values[None, None, None, None, :, None]
    )
    imag = np.zeros_like(real)

    tau = (
        np.ones((num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time))
        * (path_values * 1e-6)[None, None, None, None, :, None]
    )
    phi_r = np.zeros_like(tau)
    theta_r = np.zeros_like(tau)
    phi_t = np.ones_like(tau) * 0.5
    theta_t = np.ones_like(tau) * -0.25

    doppler = None
    if include_doppler:
        doppler = np.ones_like(tau) * 10.0

    return FakePaths(
        a=(real, imag),
        tau=tau,
        phi_r=phi_r,
        theta_r=theta_r,
        phi_t=phi_t,
        theta_t=theta_t,
        doppler=doppler,
    )


def _reduce_expected(arr):
    reduced = np.mean(arr, axis=(1, 3, 5))
    reduced = reduced[np.newaxis, ...]
    if reduced.ndim > 3:
        reduced = np.mean(reduced, axis=2)
    return reduced


def test_rt_extractor_paths_reduction(monkeypatch):
    """Ensure RTFeatureExtractor reduces Sionna Paths to expected shapes/values."""
    monkeypatch.setattr(features_module, "SIONNA_AVAILABLE", True)
    extractor = RTFeatureExtractor()
    paths = _make_paths(include_doppler=True)

    rt_features = extractor.extract(paths)

    expected_gains = _reduce_expected(np.abs(paths.a[0] + 1j * paths.a[1]))
    expected_delays = _reduce_expected(paths.tau)

    assert rt_features.is_mock is False
    assert rt_features.path_gains.shape == (1, 1, 3)
    assert rt_features.path_delays.shape == (1, 1, 3)
    np.testing.assert_allclose(rt_features.path_gains, expected_gains)
    np.testing.assert_allclose(rt_features.path_delays, expected_delays)
    assert np.all(rt_features.num_paths == 3)


def test_rt_extractor_missing_doppler(monkeypatch):
    """Verify missing doppler falls back to zeros."""
    monkeypatch.setattr(features_module, "SIONNA_AVAILABLE", True)
    extractor = RTFeatureExtractor()
    paths = _make_paths(include_doppler=False)

    rt_features = extractor.extract(paths)

    assert rt_features.path_doppler.shape == rt_features.path_delays.shape
    assert np.allclose(rt_features.path_doppler, 0.0)


def test_radio_map_generator_requires_sionna(monkeypatch, tmp_path):
    """RadioMapGenerator should fail fast when Sionna is unavailable."""
    monkeypatch.setattr(radio_map_generator, "SIONNA_AVAILABLE", False)

    with pytest.raises(ImportError):
        radio_map_generator.RadioMapGenerator(
            radio_map_generator.RadioMapConfig(output_dir=tmp_path)
        )
