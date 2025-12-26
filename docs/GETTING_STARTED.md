# Getting Started 🚀

Get up and running with the UE Localization pipeline in under 10 minutes.

---

## Prerequisites

- **Python 3.10+**
- **CUDA GPU** (recommended for Sionna ray tracing)
- **~10 GB disk space** for datasets

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Nirtzur0/CellularPositioningResearch.git
cd CellularPositioningResearch

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Test

Run a minimal end-to-end pipeline to verify everything works:

```bash
python run_pipeline.py --quick-test
```

This will:
1. Generate a small scene (Boulder, CO area)
2. Create synthetic radio measurements
3. Train for 5 epochs
4. Output results to `checkpoints/quick_test/`

Expected runtime: **5-10 minutes** on a modern laptop.

---

## Full Pipeline

For production training across multiple cities:

```bash
# Run the full experiment script
bash run_full_experiment.sh
```

Or customize with CLI flags:

```bash
python run_pipeline.py \
    --scene-config configs/scene_generation/scene_generation.yaml \
    --data-config configs/data_generation/data_generation_sionna.yaml \
    --epochs 100 \
    --batch-size 32 \
    --comet  # Enable Comet ML logging
```

---

## Directory Structure

```
CellularPositioningResearch/
├── configs/              # YAML configuration files
│   ├── data_generation/  # Data pipeline configs
│   ├── scene_generation/ # Scene generation configs
│   └── training/         # Model training configs
├── data/
│   ├── scenes/           # Generated Sionna scenes
│   └── processed/        # Zarr datasets
├── src/
│   ├── data_generation/  # M2: Feature extraction
│   ├── datasets/         # PyTorch datasets
│   ├── models/           # M3: Transformer architecture
│   ├── physics_loss/     # M4: Differentiable physics
│   ├── scene_generation/ # M1: Scene creation
│   └── training/         # Lightning training module
├── web/                  # M5: Streamlit interface
├── tests/                # Pytest test suite
└── docs/                 # This documentation
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/training/training.yaml` | Model architecture & training params |
| `configs/data_generation/data_generation_sionna.yaml` | Data generation settings |
| `configs/scene_generation/scene_generation.yaml` | Scene creation params |

---

## Web Interface

Launch the Streamlit visualization app:

```bash
streamlit run web/app.py
```

Open [http://localhost:8501](http://localhost:8501) to:
- Visualize ground truth vs predictions
- Analyze error distributions
- Inspect radio and OSM maps

---

## Next Steps

1. **Read the [Architecture](ARCHITECTURE.md)** for technical details
2. **Check [Milestones](MILESTONES.md)** for implementation status
3. **Explore [Sionna](SIONNA.md)** for ray-tracing integration
4. **Run [Pipeline](PIPELINE.md)** for CLI reference
