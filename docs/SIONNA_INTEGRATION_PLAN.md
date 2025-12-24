# Sionna Integration Plan (In Progress)

**Updated:** January 2026  
**Status:** In progress — code paths exist, real runs still need validation  
**Goal:** Move data generation from mock-only to validated Sionna RT+PHY+SYS output

---

## Where We Are
- ✅ Scene generation produces Mitsuba XML + metadata (WGS84 + UTM bounds) via our in-repo Geo2SigMap fork.
- ✅ MultiLayerDataGenerator runs end-to-end in mock mode and writes stacked features.
- ✅ Sionna hooks are present: scene loading, TX/RX setup, path computation, and feature extraction can run if Sionna is installed.
- ⚠️ No recorded end-to-end run with real Sionna yet; defaults effectively stay in mock unless Sionna/TF are available.
- ⚠️ Antenna/ray settings and performance for sub-6/mmWave are not yet tuned or documented.

See `docs/SIONNA_STATUS.md` for the live status and quick instructions.

---

## Immediate Validation Steps
1) **Smoke-test real Sionna on one tiny scene**
   - Input: a single scene from `data/scenes`, `num_samples=1_000_000`, `max_depth=3`, diffraction off.
   - Expectation: `_simulate_measurement` returns non-mock RT features (no fallback logged).
   - Output: log timing + sample stats to capture performance baseline.

2) **Tighten defaults by band**
   - Sub-6: 8x8 planar, 0.5λ spacing, ~10° downtilt.
   - mmWave: 16x16 planar, consider narrower beams; start with smaller ray counts for speed.
   - Lock these in `configs/data_generation/data_generation_sionna.yaml`.

3) **User-facing toggle**
   - Add a CLI/config flag for `use_mock_mode` in the dataset script.
   - Fail fast with a clear message when Sionna/TF are missing but real mode is requested.

4) **Regression safety**
   - Add a tiny Mitsuba fixture and a test that asserts non-mock outputs when Sionna is installed (skip otherwise).
   - Improve logging around `_simulate_measurement` to surface Sionna failures and fallbacks.

---

## Key Entry Points
- `src/data_generation/multi_layer_generator.py`: `_load_sionna_scene`, `_setup_transmitters`, `_setup_receiver`, `_simulate_measurement`.
- `src/data_generation/features.py`: `RTFeatureExtractor.extract` + PHY/MAC extraction (real + mock).
- `configs/data_generation/data_generation_sionna.yaml`: Sionna-specific knobs (ray counts, antennas, sampling).
- `tests/test_sionna_integration.py`: Smoke tests (currently mock-focused).

---

## Definition of Done
- Real Sionna run completes on at least one scene with logged timing and sample stats.
- Defaults in `configs/data_generation/data_generation_sionna.yaml` produce stable results for sub-6 and mmWave presets.
- Dataset script exposes a clear “mock vs. Sionna” switch with graceful errors.
- Regression test in place (skipped if Sionna absent) that verifies non-mock outputs.

