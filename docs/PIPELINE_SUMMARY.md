# End-to-End Pipeline Summary

## Created Files

1. **`run_pipeline.py`** - Main orchestration script (Python)
   - Coordinates all pipeline steps
   - Configurable parameters
   - Automatic config generation
   - Error handling and logging
   - Progress reporting

2. **`run_pipeline.sh`** - Shell wrapper (Bash)
   - Environment activation
   - Dependency checks
   - Colored output
   - Simple error handling

3. **`PIPELINE.md`** - Complete documentation
   - Usage examples
   - Parameter reference
   - Troubleshooting guide
   - Performance tips

4. **`Makefile`** - Convenient shortcuts
   - `make quick-test` - Quick validation
   - `make pipeline` - Full run
   - `make setup` - Install dependencies
   - `make clean` - Remove outputs

## Usage Examples

### Simplest (One Command)
```bash
./run_pipeline.sh --quick-test
```

### With Make (if available)
```bash
make quick-test
```

### Full Control
```bash
python run_pipeline.py \
    --bbox -105.28 40.014 -105.27 40.020 \
    --num-tx 5 \
    --num-trajectories 200 \
    --num-ues 100 \
    --epochs 50 \
    --batch-size 32 \
    --wandb \
    --run-name "experiment_v1"
```

### Skip Already-Done Steps
```bash
# Already have scenes
./run_pipeline.sh --skip-scenes

# Already have dataset
./run_pipeline.sh --skip-scenes --skip-dataset

# Training only
./run_pipeline.sh --train-only
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│                   run_pipeline.py                       │
│              (Orchestration & Coordination)             │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   STEP 1     │    │   STEP 2     │    │   STEP 3     │
│              │    │              │    │              │
│   Generate   │───▶│   Generate   │───▶│    Train     │
│   Scenes     │    │   Dataset    │    │    Model     │
│              │    │              │    │              │
│ (OSM + TX)   │    │  (Sionna)    │    │ (PyTorch)    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
  data/scenes/      data/processed/      checkpoints/
```

## Configuration

### Quick Test Parameters (--quick-test)
- Area: ~300m x 300m
- Transmitters: 2
- Trajectories: 50
- UEs per trajectory: 20
- Training epochs: 5
- Duration: ~10-20 minutes

### Default Parameters
- Area: Boulder, CO (small section)
- Transmitters: 3
- Trajectories: 100
- UEs per trajectory: 50
- Training epochs: 10
- Duration: ~30-60 minutes

### Production Parameters (Example)
```bash
./run_pipeline.sh \
    --bbox -105.30 40.00 -105.20 40.05 \
    --num-tx 20 \
    --num-trajectories 1000 \
    --num-ues 200 \
    --epochs 200 \
    --batch-size 64 \
    --wandb \
    --run-name "production_v1"
```
Duration: Several hours to days

## Output Structure

```
transformer-ue-localization/
├── data/
│   ├── scenes/
│   │   └── <scene-name>/
│   │       └── scene_*/              # 3D scenes
│   │
│   └── processed/
│       └── <scene-name>_dataset/
│           └── dataset_*.zarr        # Training data
│
├── checkpoints/
│   └── <run-name>/
│       ├── pipeline_report.json      # Execution summary
│       ├── training_config.yaml      # Generated config
│       └── *.ckpt                    # Model weights
│
└── logs/
    └── pipeline_*.log                # Detailed logs
```

## Key Features

### ✅ Complete Automation
- Single command runs entire pipeline
- Automatic dependency management
- Progress tracking and logging

### ✅ Flexible Control
- Skip completed steps
- Resume from checkpoints
- Override any parameter

### ✅ Error Handling
- Stops on first error
- Detailed error messages
- Preserves partial progress

### ✅ Reproducibility
- Saves all configuration
- Generates execution report
- Timestamps everything

### ✅ Development Friendly
- Quick test mode for validation
- Clean commands for fresh runs
- Integrates with existing workflows

## Common Workflows

### First Time Setup
```bash
# 1. Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# 2. Test everything works
./run_pipeline.sh --quick-test

# 3. Run on your area of interest
./run_pipeline.sh --bbox <your-coords> --num-tx 5 --epochs 50
```

### Iterative Development
```bash
# Generate scenes once
./run_pipeline.sh --bbox -105.28 40.014 -105.27 40.020

# Experiment with dataset parameters
./run_pipeline.sh --skip-scenes --num-trajectories 200
./run_pipeline.sh --skip-scenes --num-trajectories 500
./run_pipeline.sh --skip-scenes --num-ues 200

# Try different training configs
./run_pipeline.sh --skip-scenes --skip-dataset --epochs 20 --batch-size 8
./run_pipeline.sh --skip-scenes --skip-dataset --config configs/training/training.yaml
```

### Production Run
```bash
# Clean everything and start fresh
./run_pipeline.sh --clean --wandb --run-name "production_$(date +%Y%m%d)"
```

## Next Steps

1. **Run quick test**: `./run_pipeline.sh --quick-test`
2. **Check outputs**: Review generated scenes, dataset, and checkpoints
3. **Adjust parameters**: Modify based on your requirements
4. **Scale up**: Run with production parameters
5. **Monitor training**: Use W&B or Lightning logs

## Support

- **Documentation**: See [PIPELINE.md](PIPELINE.md)
- **Help**: `./run_pipeline.sh --help` or `python run_pipeline.py --help`
- **Examples**: Check PIPELINE.md "Examples" section
- **Troubleshooting**: See PIPELINE.md "Troubleshooting" section
