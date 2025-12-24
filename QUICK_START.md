# Quick Start: One-Stop Pipeline

This guide uses the unified entry point: `run_pipeline.py`.

## Prerequisites

```bash
cd /home/ubuntu/projects/CellularPositioningResearch
pip install -r requirements.txt
```

## One-Command Pipeline

### Quick test (recommended first run)
```bash
python run_pipeline.py --quick-test
```

### Full pipeline
```bash
python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

### Train on multiple locations, evaluate on another
```bash
python run_pipeline.py \
  --train-datasets data/processed/paris.zarr data/processed/tokyo.zarr \
  --eval-dataset data/processed/nyc.zarr
```

When `--eval-dataset` (or `--eval-data-config`) is provided, the pipeline runs a dedicated evaluation step on that held‑out dataset.

### Light training (fast sanity check)
```bash
python run_pipeline.py --quick-test --light-train
```

### Hyperparameter optimization (Optuna)
```bash
# Install optional deps
pip install optuna optuna-integration

# Run Optuna study (then train + eval with best params)
python run_pipeline.py --optimize --n-trials 20 --study-name ue-localization
```

### Skip steps if already done
```bash
# Use existing scenes
python run_pipeline.py --skip-scenes

# Use existing dataset
python run_pipeline.py --skip-scenes --skip-dataset

# Training only
python run_pipeline.py --train-only

# Evaluation only (requires --eval-dataset or --eval-data-config)
python run_pipeline.py --eval-only --eval-dataset data/processed/nyc.zarr
```

## What the pipeline does

1. Generate scenes (OSM -> Mitsuba XML)
2. Generate dataset (RT/PHY/MAC features -> Zarr)
3. Train model (transformer)
4. Evaluate on held-out dataset (optional)
5. Write pipeline report

Outputs:
- Scenes: `data/scenes/<scene_name>/`
- Datasets: `data/processed/*.zarr`
- Checkpoints + configs: `checkpoints/<run_name>/`

## Advanced: Run individual steps

### Step 1: Generate scenes
```bash
python scripts/scene_generation/generate_scenes.py \
  --bbox -105.28 40.014 -105.27 40.020 \
  --output data/scenes/boulder_test \
  --num-tx 3 \
  --site-strategy grid
```

### Step 2: Generate dataset
```bash
python scripts/generate_dataset.py \
  --scene-dir data/scenes/boulder_test \
  --output-dir data/processed \
  --num-ue 100 \
  --num-reports 10 \
  --carrier-freq 3.5e9
```

### Step 3: Train
```bash
python scripts/train.py \
  --config configs/training/training_simple.yaml \
  --run-name boulder_test_v1
```

### Step 4: Evaluate
```bash
python scripts/train.py \
  --config configs/training/training_simple.yaml \
  --resume checkpoints/last.ckpt \
  --test-only
```

## Configs

- `configs/scene_generation/scene_generation.yaml`
- `configs/data_generation/data_generation.yaml`
- `configs/data_generation/data_generation_sionna.yaml` (optional)
- `configs/training/training_simple.yaml`

For more options:
```bash
python run_pipeline.py --help
```
