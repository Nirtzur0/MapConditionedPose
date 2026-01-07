# Pipeline

`run_pipeline.py` is the main entry point that runs the complete pipeline: scene generation → data generation → training.

## Quick Start

```bash
# Quick test (uses existing scenes, minimal data, 3 epochs)
python run_pipeline.py --quick-test

# Full run with custom name
python run_pipeline.py --name my_experiment

# Skip specific steps
python run_pipeline.py --skip-scenes  # Use existing scenes
python run_pipeline.py --skip-training  # Generate data only
```

## Configuration

The pipeline uses a dataclass-based configuration with sensible defaults. Key settings:

- **Scenes**: Cities to generate, number of transmitters, site placement strategy
- **Data**: Carrier frequency, bandwidth, UE density, train/val/test split ratios  
- **Training**: Epochs, batch size, learning rate

## Output Structure

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

## From YAML Config

```bash
python run_pipeline.py --config my_config.yaml
```

## Environment Variables

- `COMET_API_KEY`: Enable Comet ML experiment tracking
- `OVERPASS_URL`: Custom OSM Overpass API endpoint
