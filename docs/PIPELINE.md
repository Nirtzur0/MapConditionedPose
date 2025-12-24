# Pipeline Orchestration

`run_pipeline.py` is the single CLI that runs scenes dataset training eval.

## Quick start

```bash
python run_pipeline.py --quick-test
```

## Full run

```bash
python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

## Multi-location training + held-out eval

```bash
python run_pipeline.py \
 --train-datasets data/processed/paris.zarr data/processed/tokyo.zarr \
 --eval-dataset data/processed/nyc.zarr
```

## Optuna (optional)

```bash
pip install optuna optuna-integration
python run_pipeline.py --optimize --n-trials 20 --study-name ue-localization
```

## Common patterns

```bash
# Skip scenes
python run_pipeline.py --skip-scenes

# Skip scenes + dataset
python run_pipeline.py --skip-scenes --skip-dataset

# Train only
python run_pipeline.py --train-only

# Eval only
python run_pipeline.py --eval-only --eval-dataset data/processed/nyc.zarr

# Light training
python run_pipeline.py --quick-test --light-train
```

## Key flags

- Scene: `--bbox`, `--scene-name`, `--num-tx`, `--site-strategy`
- Dataset: `--carrier-freq`, `--bandwidth`, `--num-ues`, `--num-trajectories`
- Training: `--config`, `--epochs`, `--batch-size`, `--learning-rate`
- Multi-dataset: `--train-datasets`, `--train-data-configs`, `--eval-dataset`, `--eval-data-config`
- Control: `--skip-scenes`, `--skip-dataset`, `--skip-training`, `--skip-eval`, `--train-only`, `--eval-only`, `--clean`

## Outputs

- Scenes: `data/scenes/<scene_name>/`
- Datasets: `data/processed/*.zarr`
- Checkpoints + configs: `checkpoints/<run_name>/`
- Logs: `logs/`

For the full flag list:
```bash
python run_pipeline.py --help
```
