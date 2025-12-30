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
# Quick test (recommended for first run)
python run_pipeline.py --quick-test

# Full pipeline
python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

### Hyperparameter Optimization (Optuna) 🔎

```bash
# Install optional deps
pip install optuna optuna-integration

# Run Optuna study (then train + eval with best params)
python run_pipeline.py --optimize --n-trials 20 --study-name ue-localization
```

See `QUICK_START.md` for the compact workflow and `docs/PIPELINE.md` for details.

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
M1: OSM -> Scene Builder -> Mitsuba scenes
M2: Sionna RT -> RT/PHY/MAC features -> Zarr
M3: Transformer (radio + map encoders -> fusion -> position)
M4: Training + evaluation (+ optional physics loss)
M5: Web UI for inspection and demos
```

## Documentation 📚

- [QUICK_START.md](QUICK_START.md) - Short start guide
- [docs/paper/paper.tex](docs/paper/paper.tex) - **Research Paper: Physics-Informed Transformer for UE Localization**
- [docs/README.md](docs/README.md) - Documentation index
- [docs/PIPELINE.md](docs/PIPELINE.md) - Pipeline usage and options
- [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Design + milestones
- [docs/SYSTEM_INTEGRATION_GUIDE.md](docs/SYSTEM_INTEGRATION_GUIDE.md) - Data flow and integration
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Current status and roadmap

## Key Directories

```
configs/ # Pipeline configs
scripts/ # Dataset + training scripts
src/ # Core code
tests/ # Test suite
web/ # Streamlit app
```
