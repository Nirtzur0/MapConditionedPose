# Quick Start

Use the unified CLI: `run_pipeline.py`.

## Install

```bash
cd /home/ubuntu/projects/CellularPositioningResearch
pip install -r requirements.txt
```

## Run the pipeline

```bash
# Quick test
python run_pipeline.py --quick-test

# Full run
python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

## Multi-location training + held-out eval

```bash
python run_pipeline.py \
 --train-datasets data/processed/paris.zarr data/processed/tokyo.zarr \
 --eval-dataset data/processed/nyc.zarr
```

## Light training (fast sanity check)

```bash
python run_pipeline.py --quick-test --light-train
```

## Optuna (optional)

```bash
pip install optuna optuna-integration
python run_pipeline.py --optimize --n-trials 20 --study-name ue-localization
```

## Skip steps

```bash
python run_pipeline.py --skip-scenes
python run_pipeline.py --skip-scenes --skip-dataset
python run_pipeline.py --train-only
python run_pipeline.py --eval-only --eval-dataset data/processed/nyc.zarr
```

## Outputs

- Scenes: `data/scenes/<scene_name>/`
- Datasets: `data/processed/*.zarr`
- Checkpoints + configs: `checkpoints/<run_name>/`

For full options:
```bash
python run_pipeline.py --help
```
