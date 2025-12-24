# Sionna Integration - Status Correction

**Updated:** January 2026  
**Status:** Not fully complete — real Sionna path exists, validation pending  
**Next Step:** Run a real Sionna dataset as outlined in `docs/SIONNA_INTEGRATION_PLAN.md`

---

## Quick Summary
- The code contains Sionna hooks (scene loading, TX/RX setup, path computation, feature extraction) alongside mock fallbacks.
- We do **not** yet have a logged, successful end-to-end run with real Sionna; current tests are mock-focused.
- Use this file as a correction to the earlier “complete” claim. For the live view, read `docs/SIONNA_STATUS.md`.

## How to Run Today
- **Mock mode (recommended until validation):** set `use_mock_mode: true` (or leave default when Sionna isn’t installed) and run data generation to exercise the full pipeline quickly.
- **Attempt real mode:** install Sionna + TensorFlow + Dr.Jit, set `use_mock_mode: false`, start with small ray counts (`num_samples: 1_000_000`, `max_depth: 3`, diffraction off) on a tiny scene; expect to iterate on antenna presets.
- Inputs: scenes from `data/scenes/` with metadata index `data/scenes/metadata.json` produced by scene generation.

## Files to Know
- Code: `src/data_generation/features.py`, `src/data_generation/multi_layer_generator.py`
- Config: `configs/data_generation/data_generation_sionna.yaml`
- Tests: `tests/test_sionna_integration.py` (mock-centric; add real-Sionna regression after first success)

## Definition of Done
- Real Sionna run completes on at least one scene with timing and sample stats captured.
- Defaults in `configs/data_generation/data_generation_sionna.yaml` are validated for sub-6 and mmWave presets.
- Dataset script exposes a clear “mock vs. Sionna” switch with graceful errors when Sionna is missing.
- Regression test (skipped when Sionna absent) verifies non-mock outputs using a tiny Mitsuba fixture.

---

For current status and the step-by-step plan, see `docs/SIONNA_STATUS.md` and `docs/SIONNA_INTEGRATION_PLAN.md`.
