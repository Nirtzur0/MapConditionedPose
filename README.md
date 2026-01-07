# Transformer-Based UE Localization 🚀

![Tests](https://img.shields.io/badge/tests-89%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

A practical end-to-end pipeline for 5G NR UE localization using multi-layer radio measurements and a transformer model.

## Overview

The pipeline learns to localize UEs using:
- **Layer 1 (RT)**: propagation paths (gains, AoA/AoD, delays)
- **Layer 2 (PHY/FAPI)**: link measurements (RSRP, RSRQ, CQI, RI)
- **Layer 3 (MAC/RRC)**: system features (cell IDs, timing advance, throughput)

## Project Status

| Milestone | Status | Notes |
|-----------|--------|------|
| M1: Scene Generation | Complete | OSM to Mitsuba scenes |
| M2: Data Generation | Complete | RT/PHY/MAC features to Zarr |
| M3: Transformer Model | Complete | Dual-encoder + fusion |
| M4: Training Pipeline | Complete | Training + eval + Optuna |
| M5: Web UI | Complete | Streamlit demo app |

## Quick Start ⚡

```bash
# Quick test (uses existing scenes, minimal data, 3 epochs)
python run_pipeline.py --quick-test

# Full run with custom name
python run_pipeline.py --name my_experiment

# Skip specific steps
python run_pipeline.py --skip-scenes  # Use existing scenes
python run_pipeline.py --skip-training  # Generate data only

# From config file
python run_pipeline.py --config my_config.yaml
```

### Output Structure

Each run creates organized outputs:
```
outputs/{experiment_name}/
├── config.yaml          # Saved configuration
├── scenes/              # Generated 3D scenes
├── data/
│   ├── dataset_train_*.zarr
│   ├── dataset_val_*.zarr
│   └── dataset_test_*.zarr
├── checkpoints/         # Model checkpoints
└── report.yaml          # Final metrics
```

See [docs/PIPELINE.md](docs/PIPELINE.md) for detailed usage.

## Manual Installation

```bash
# Clone repository
git clone <repository-url>
cd transformer-ue-localization

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Tests 🧪

```bash
pytest tests/ -v
```

## Architecture (High-Level)

```
Scene Generation: OSM → Scene Builder → 3D Mitsuba scenes
Data Generation:  Sionna RT → RT/PHY/MAC features → Zarr datasets (train/val/test)
Model Training:   Transformer (radio + map encoders → fusion → position)
Evaluation:       Test metrics and visualization
```

**Single Entry Point**: `run_pipeline.py` orchestrates the entire pipeline with direct function calls.

## Documentation 📚

- [docs/PIPELINE.md](docs/PIPELINE.md) - **Main pipeline usage and CLI options**
- [docs/README.md](docs/README.md) - Documentation index
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - Setup and installation
- [docs/paper/](docs/paper/) - Research paper and figures

## Key Directories

```
configs/         # Model and training configurations
scripts/         # Standalone utility scripts
src/            # Core source code
  ├── models/         # Transformer model architecture
  ├── training/       # Training and evaluation logic
  ├── data_generation/# Data generation from scenes
  ├── scene_builder/  # OSM to 3D scene conversion
  └── datasets/       # Data loading and preprocessing
tests/          # Test suite
web/            # Streamlit demo app
run_pipeline.py # Main entry point
```
