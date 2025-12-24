# Pipeline Orchestration

Complete end-to-end pipeline orchestration for the Transformer UE Localization project.

## Quick Start

### Option 1: Quick Test (Recommended for first run)
```bash
./run_pipeline.sh --quick-test
```

This runs a minimal pipeline:
- Small geographic area (~300m x 300m)
- 2 transmitters
- 50 trajectories with 20 UEs each
- 5 training epochs
- Completes in ~10-20 minutes

### Option 2: Full Pipeline
```bash
./run_pipeline.sh --bbox -105.28 40.014 -105.27 40.020 --num-tx 5 --epochs 50
```

### Option 3: Direct Python
```bash
source venv/bin/activate
python run_pipeline.py --help
```

## Pipeline Steps

The orchestrator runs 4 main steps:

1. **Scene Generation**: Creates 3D scenes with buildings, terrain, and transmitter sites
2. **Dataset Generation**: Simulates radio propagation and generates training data
3. **Model Training**: Trains the transformer model for UE localization
4. **Report Generation**: Creates execution summary and metrics

## Common Usage Patterns

### Skip Already-Completed Steps
```bash
# Use existing scenes, regenerate dataset and train
./run_pipeline.sh --skip-scenes

# Use existing dataset, just train
./run_pipeline.sh --skip-scenes --skip-dataset

# Training only
./run_pipeline.sh --train-only
```

### Custom Parameters
```bash
# Larger area with more transmitters
./run_pipeline.sh \
    --bbox -105.30 40.00 -105.20 40.05 \
    --num-tx 10 \
    --num-trajectories 500 \
    --num-ues 100 \
    --epochs 100 \
    --batch-size 32

# Use custom training config
./run_pipeline.sh --config configs/training.yaml
```

### Resume Training
```bash
./run_pipeline.sh \
    --skip-scenes \
    --skip-dataset \
    --resume-checkpoint checkpoints/last.ckpt
```

### Clean Previous Outputs
```bash
# Remove old outputs before running
./run_pipeline.sh --clean --quick-test
```

### Enable Wandb Logging
```bash
./run_pipeline.sh --wandb --run-name "experiment_v1"
```

## Parameters

### Scene Generation
- `--bbox WEST SOUTH EAST NORTH`: Geographic bounding box
- `--scene-name NAME`: Output directory name (default: boulder_test)
- `--num-tx N`: Number of transmitters (default: 3)
- `--site-strategy`: Placement strategy (grid/random/cluster)
- `--tiles`: Generate multiple tiles

### Dataset Generation
- `--carrier-freq HZ`: Carrier frequency in Hz (default: 3.5e9)
- `--bandwidth HZ`: Bandwidth in Hz (default: 100e6)
- `--num-ues N`: UEs per trajectory (default: 50)
- `--num-trajectories N`: Number of trajectories (default: 100)

### Training
- `--config PATH`: Training config file
- `--epochs N`: Training epochs (default: 10)
- `--batch-size N`: Batch size (default: 16)
- `--learning-rate LR`: Learning rate (default: 0.0001)
- `--num-workers N`: DataLoader workers (default: 2)

### Control
- `--skip-scenes`: Skip scene generation
- `--skip-dataset`: Skip dataset generation
- `--skip-training`: Skip training
- `--train-only`: Only train (skip scenes & dataset)
- `--clean`: Remove previous outputs

### Logging
- `--wandb`: Enable W&B logging
- `--run-name NAME`: Run identifier

## Output Structure

After running, outputs are organized as:

```
data/
  scenes/
    <scene-name>/
      scene_*/          # Generated scenes
      
  processed/
    <scene-name>_dataset/
      *.zarr            # Training datasets
      
checkpoints/
  <run-name>/
    pipeline_report.json   # Execution summary
    training_config.yaml   # Generated config
    *.ckpt                 # Model checkpoints
```

## Logs

Each run generates:
- **Console output**: Real-time progress
- **Log file**: `logs/pipeline_YYYYMMDD_HHMMSS.log`
- **Pipeline report**: JSON summary in checkpoint directory

## Error Handling

The pipeline will:
- Stop on first error
- Save partial progress
- Log detailed error messages
- Provide actionable error reports

Check the log file for detailed diagnostics.

## Performance Tips

### CPU-Only Training
- Use smaller batch sizes (8-16)
- Reduce model complexity in config
- Lower number of workers (2-4)

### With GPU
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Increase batch size (32-64)
- Use more workers (8-16)
- Enable mixed precision in config

### Memory Optimization
- Reduce `num_ues` and `num_trajectories`
- Use smaller `img_size` in config
- Decrease `num_workers` if OOM errors occur

## Troubleshooting

### Scene Generation Fails
- Check internet connection (downloads OSM data)
- Verify bounding box coordinates are valid
- Ensure sufficient disk space

### Dataset Generation Slow
- Reduce `num_trajectories` or `num_ues`
- Check if Sionna/TensorFlow is using CPU only

### Training Issues
- Verify dataset path exists and is valid Zarr format
- Check model config matches dataset dimensions
- Monitor GPU/CPU memory usage

### ImportError
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Examples

### Development Workflow
```bash
# 1. Quick test to verify everything works
./run_pipeline.sh --quick-test

# 2. Generate scenes for your area of interest
./run_pipeline.sh --bbox -105.28 40.014 -105.27 40.020 --num-tx 5

# 3. Experiment with dataset parameters
./run_pipeline.sh --skip-scenes --num-trajectories 200 --num-ues 100

# 4. Run full training
./run_pipeline.sh --skip-scenes --skip-dataset --epochs 100 --wandb
```

### Production Run
```bash
./run_pipeline.sh \
    --bbox -105.30 40.00 -105.20 40.05 \
    --num-tx 20 \
    --num-trajectories 1000 \
    --num-ues 200 \
    --epochs 200 \
    --batch-size 64 \
    --wandb \
    --run-name "production_v1" \
    --clean
```

### Debugging
```bash
# Run with verbose logging
python run_pipeline.py --quick-test 2>&1 | tee logs/debug.log

# Test individual steps
python scripts/generate_scenes.py --area "Boulder, CO" --output data/scenes/test
python scripts/generate_dataset.py --scene-dir data/scenes/test --output-dir data/test
python scripts/train.py --config configs/training_simple.yaml
```

## Integration with Existing Workflows

The orchestrator is designed to work alongside manual workflows:

1. **Manual scene generation** → Use `--skip-scenes`
2. **Existing dataset** → Use `--skip-dataset`
3. **Custom training script** → Use `--skip-training`

Simply point to your existing artifacts with `--scene-name` and the orchestrator will use them.
