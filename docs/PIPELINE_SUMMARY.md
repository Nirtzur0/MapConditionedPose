# Pipeline Summary

`run_pipeline.py` is the single entry point for scenes dataset training eval.

## Quick commands

```bash
python run_pipeline.py --quick-test
python run_pipeline.py --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

## Multi-location training + held-out eval

```bash
python run_pipeline.py \
 --train-datasets data/processed/paris.zarr data/processed/tokyo.zarr \
 --eval-dataset data/processed/nyc.zarr
```

## Skip steps

```bash
python run_pipeline.py --skip-scenes
python run_pipeline.py --skip-scenes --skip-dataset
python run_pipeline.py --train-only
python run_pipeline.py --eval-only --eval-dataset data/processed/nyc.zarr
```

## Optuna (optional)

```bash
pip install optuna optuna-integration
python run_pipeline.py --optimize --n-trials 20 --study-name ue-localization
```

## Outputs

- `data/scenes/<scene_name>/`
- `data/processed/*.zarr`
- `checkpoints/<run_name>/`

For full options:
```bash
python run_pipeline.py --help
```
