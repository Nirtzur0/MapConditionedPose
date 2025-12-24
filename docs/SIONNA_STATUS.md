# Sionna Integration Status (RT + PHY + SYS)

**Updated:** January 2026  
**Status:** Partial integration with mock fallback  
**TL;DR:** The code paths for real Sionna are in place, but the system still runs in mock mode unless you have Sionna installed and wired up to real scenes. Use this as the single source of truth for where things stand and what to do next.

---

## What Works Today
- Scene generation (M1) writes Mitsuba XML + metadata with both WGS84 and UTM meter bounds (`data/scenes/<scene>/scene.xml`, `metadata.json`).
- Data generator can load scene metadata and produce mock RT/PHY/MAC features (fast, no TensorFlow/Dr.Jit needed).
- Sionna hooks exist: `RTFeatureExtractor.extract`, `MultiLayerDataGenerator._simulate_measurement`, `_setup_transmitters/_setup_receiver`, and `_load_sionna_scene` will run real ray tracing when Sionna is installed.
- Config scaffolding exists (`configs/data_generation/data_generation_sionna.yaml`) and smoke tests exercise imports and mock mode (`tests/test_sionna_integration.py`).

## What’s Still Missing / Needs Validation
- End-to-end run with real Sionna: no recorded successful run with actual scenes; defaults still fall back to mock when Sionna isn’t present.
- Performance/quality tuning: ray sampling, diffraction settings, and antenna patterns need empirical validation per band.
- Robust error handling: need clearer errors when scenes or TF/Sionna aren’t available, and better logging of path extraction failures.
- Documentation drift: earlier “Implementation Complete” note was aspirational; this file replaces that claim.

## How to Use It Right Now
1) Generate scenes with `scripts/scene_generation/generate_scenes.py` (or TileGenerator) → outputs to `data/scenes/` plus `metadata.json` index.  
2) Run data generation in mock mode for speed:
   - Set `use_mock_mode: true` in your config or pass `use_mock_mode=True` to `DataGenerationConfig`.
   - Point `scene_dir` and `scene_metadata_path` to `data/scenes` and `data/scenes/metadata.json`.
3) To attempt real Sionna:
   - Install Sionna + TensorFlow + Dr.Jit; ensure `sionna` imports succeed.
   - Set `use_mock_mode: false` and keep ray counts small (`num_samples: 1_000_000`, `max_depth: 3-5`) for a smoke test.
   - Expect to tweak antenna definitions in `_setup_transmitters/_setup_receiver` for your band (sub-6 vs. mmWave).

## Next Steps to Reach “Production”
1) Validate one small scene end-to-end with real Sionna (no diffraction, small sample count) and record timings/quality.  
2) Add graceful fallbacks and clearer logs when Sionna/TF isn’t present.  
3) Tune antenna patterns and ray settings per band; document tested presets.  
4) Wire a minimal “Sionna on/off” CLI flag into the dataset script for convenience.  
5) Add a regression test that loads a tiny Mitsuba scene and asserts non-mock outputs when Sionna is installed.

## Key Files (Sionna Path)
- `src/data_generation/features.py`: RT/PHY/MAC feature extraction (real + mock).
- `src/data_generation/multi_layer_generator.py`: Scene loading, UE sampling, Sionna simulation loop.
- `configs/data_generation/data_generation_sionna.yaml`: Sionna-specific knobs (ray counts, antenna config, sampling).
- `tests/test_sionna_integration.py`: Smoke tests (currently mock-centric).

## Contact / Ownership
- Data generation & Sionna integration: `src/data_generation/*`
- Scene generation (inputs): `src/scene_generation/*`

