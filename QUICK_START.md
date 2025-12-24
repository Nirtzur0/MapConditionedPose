# 🚀 Quick Start: Complete Workflow

## End-to-End Pipeline: Data → Training → Visualization

This guide walks you through the complete pipeline from scratch to visualizing results.

---

## 📋 Prerequisites

```bash
cd /home/ubuntu/projects/transformer-ue-localization

# 1. Install all dependencies (includes testing, training, and web interface)
pip install -r requirements.txt

# 2. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

---

## 🎯 The Complete Workflow

### **Step 1: Generate Synthetic Scenes (M1)**

Generate realistic 3D scenes from OpenStreetMap data:

```bash
# Quick test with small area (Boulder, CO)
python scripts/scene_generation/generate_scenes.py \
    --bbox 40.014 -105.28 40.020 -105.27 \
    --output data/scenes/boulder_test \
    --num-sites 3 \
    --site-strategy "grid"

# Expected output:
# data/scenes/boulder_test/
# ├── scene_001/
# │   ├── scene.xml           # Mitsuba scene for Sionna RT
# │   ├── buildings.ply       # 3D building meshes
# │   └── metadata.json       # Site locations, materials, bbox
# └── tiles_metadata.json
```

**What this does:**
- Downloads OSM data for the bounding box
- Generates 3D building meshes
- Places cell tower sites (TX antennas)
- Randomizes material properties (walls, roofs, ground)

---

### **Step 2: Generate Training Data (M2)**

Generate synthetic radio measurements with multi-layer features:

```bash
# Generate dataset from scenes
python scripts/generate_dataset.py \
    --scene-dir data/scenes/boulder_test \
    --output-dir data/processed \
    --num-ue-per-tile 1000 \
    --num-reports-per-ue 10 \
    --carrier-freq 3.5e9

# Expected output:
# data/processed/dataset.zarr/
# ├── radio_measurements/     # RT + PHY + SYS features
# ├── positions/             # Ground truth UE positions
# ├── metadata/              # Scene info, TX locations
# └── .zattrs               # Dataset attributes
```

**What this does:**
- Runs Sionna RT to compute radio propagation paths
- Extracts multi-layer features:
  - **RT Layer**: Path gains, ToA, AoA, Doppler
  - **PHY Layer**: RSRP, RSRQ, SINR, CQI
  - **SYS Layer**: Throughput, BLER, Cell IDs, Timing Advance
- Adds measurement realism (dropout, quantization, temporal gaps)
- Saves to compressed Zarr format for efficient PyTorch loading

---

### **Step 3: Train the Model (M3 + M4)**

Train the transformer-based localization model:

```bash
# Training with default config
python scripts/train.py \
    --config configs/training/training.yaml \
    --wandb-project ue-localization \
    --run-name boulder_test_v1

# Training with physics loss (optional but recommended)
python scripts/train.py \
    --config configs/training/training.yaml \
    --wandb-project ue-localization \
    --run-name boulder_test_physics

# Resume from checkpoint
python scripts/train.py \
    --config configs/training/training.yaml \
    --resume checkpoints/last.ckpt
```

**What this does:**
- Loads dataset from `data/processed/dataset.zarr`
- Trains multi-scale transformer model:
  - Radio encoder (processes measurement sequences)
  - Map encoder (processes precomputed radio/building maps)
  - Cross-attention fusion
  - Coarse head (32×32 grid classification)
  - Fine head (precise regression with uncertainty)
- Applies physics loss (optional, validates predictions against radio maps)
- Saves checkpoints to `checkpoints/`

**Training outputs:**
```
checkpoints/
├── best_model.pt          # Best validation loss
├── last.ckpt              # Latest checkpoint (for resuming)
└── epoch_*.ckpt           # Periodic checkpoints
```

---

### **Step 4: Evaluate Performance**

Test the trained model and generate metrics:

```bash
# Run test evaluation
python scripts/train.py \
    --config configs/training/training.yaml \
    --resume checkpoints/best_model.pt \
    --test-only

# Expected output:
# Test Results:
# - Median Error: 12.3m
# - 90th Percentile: 45.7m
# - Mean Error: 18.9m
# - MAE: 15.4m
```

**What this evaluates:**
- Localization accuracy (median error, percentiles)
- Error distribution (CDF, histogram)
- Per-layer feature contributions
- Uncertainty calibration

---

### **Step 5: Visualize with Streamlit App (M5)**

Launch the web interface to explore results:

```bash
# Start the web app
cd web
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.headless false

# App opens at: http://localhost:8501
```

**Web Interface Features:**

1. **📊 Overview Page**
   - System architecture diagram
   - Dataset statistics
   - Model summary

2. **📈 Metrics Dashboard**
   - Error CDF curves
   - Accuracy vs timesteps
   - Feature ablation analysis
   - Per-scene performance breakdown

3. **🔍 Live Inference**
   - Upload measurement files
   - Run model inference
   - View predictions on map
   - Trajectory visualization

4. **📉 Analysis**
   - Error distribution heatmaps
   - Attention weight visualization
   - Uncertainty quantification
   - Physics loss validation

---

## 🎓 Understanding Results

### **Good Performance Indicators**

✅ **Median Error < 20m** - Model generalizes well  
✅ **90th %ile < 50m** - Few catastrophic failures  
✅ **Uncertainty calibrated** - Predicted σ matches actual error  
✅ **Smooth trajectories** - Temporal consistency maintained  

### **Common Issues**

❌ **High median error (>50m)**
- **Cause**: Insufficient training data or scene diversity
- **Fix**: Generate more scenes with varied environments

❌ **Large error variance**
- **Cause**: Poor feature quality or overfitting
- **Fix**: Add dropout, increase data augmentation

❌ **NLOS errors**
- **Cause**: Missing multipath information
- **Fix**: Enable physics loss, increase RT layer weight

---

## 📊 Example Complete Run

Here's what a successful end-to-end run looks like:

```bash
# 1. Generate 10 scenes (5 minutes)
python scripts/scene_generation/generate_scenes.py \
    --bbox 40.01 -105.28 40.03 -105.26 \
    --output data/scenes/boulder_batch1 \
    --num-tiles 10 \
    --num-sites 5

# 2. Generate 100K training samples (30 minutes)
python scripts/generate_dataset.py \
    --scene-dir data/scenes/boulder_batch1 \
    --output-dir data/processed/boulder_100k \
    --num-ue-per-tile 10000 \
    --num-reports-per-ue 20

# 3. Train model (2-4 hours on GPU)
python scripts/train.py \
    --config configs/training/training.yaml \
    --wandb-project ue-localization \
    --run-name boulder_100k_baseline

# 4. Test evaluation (5 minutes)
python scripts/train.py \
    --config configs/training/training.yaml \
    --resume checkpoints/best_model.pt \
    --test-only

# 5. Launch visualization
cd web && streamlit run streamlit_app.py
```

**Expected Timeline:**
- Scene generation: ~30 seconds per scene
- Data generation: ~100 UEs/second (CPU), ~1000 UEs/second (GPU)
- Training: ~5-10K samples/second (depends on GPU)
- Inference: ~1000 samples/second

---

## 🔬 Demo Mode (No Training Required)

Want to see the app immediately? Use demo mode:

```bash
cd web
streamlit run streamlit_app.py

# In the app:
# 1. Go to "🔍 Live Inference" page
# 2. Click "Generate Demo Data"
# 3. Set timesteps = 10
# 4. Click "Generate Measurements"
# 5. Click "Run Inference"
```

**Demo mode:**
- Uses synthetic random measurements
- Mock model predictions (random walk)
- All visualizations functional
- Perfect for testing UI without waiting for training

---

## 🛠️ Configuration

### Key Config Files

**`configs/training/training.yaml`** - Training hyperparameters
```yaml
training:
  data:
    train_path: "data/processed/dataset.zarr"
    batch_size: 64
  
  optimizer:
    lr: 1e-4
    weight_decay: 0.01
  
  num_epochs: 100
```

**`configs/scene_generation/scene_generation.yaml`** - Scene generation
```yaml
scene:
  tile_size: 1000  # meters
  num_sites: 5
  site_strategy: "grid"

materials:
  randomize: true
  domains: ["urban", "suburban", "rural"]
```

---

## 📁 Directory Structure

After running the complete pipeline:

```
transformer-ue-localization/
├── data/
│   ├── scenes/              # M1 output
│   │   └── boulder_test/
│   │       ├── scene_001/
│   │       └── tiles_metadata.json
│   └── processed/           # M2 output
│       └── dataset.zarr/
│           ├── radio_measurements/
│           └── positions/
├── checkpoints/             # M3 output
│   ├── best_model.pt
│   └── last.ckpt
├── logs/                    # Training logs
│   └── lightning_logs/
└── web/                     # M5 app
    └── streamlit_app.py
```

---

## 🐛 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'sionna'"**
**Solution:**
```bash
pip install sionna
# Or run in mock mode (for testing without Sionna):
export MOCK_SIONNA=1
python scripts/generate_dataset.py ...
```

### **Issue: "Scene directory not found"**
**Solution:** Run M1 first to generate scenes
```bash
python scripts/scene_generation/generate_scenes.py --bbox ... --output data/scenes/test
```

### **Issue: "Dataset not found during training"**
**Solution:** Update config path or run M2 first
```yaml
# configs/training/training.yaml
training:
  data:
    train_path: "data/processed/dataset.zarr"  # Update this path
```

### **Issue: "CUDA out of memory"**
**Solution:** Reduce batch size
```yaml
# configs/training/training.yaml
training:
  data:
    batch_size: 32  # Reduce from 64
```

---

## 📚 Next Steps

1. **Scale up**: Generate more diverse scenes (urban, suburban, rural)
2. **Optimize**: Tune hyperparameters using Weights & Biases
3. **Deploy**: Export model to ONNX for production inference
4. **Integrate**: Connect to real-world measurement collection system

---

## 🔗 Related Documentation

- [SYSTEM_INTEGRATION_GUIDE.md](docs/SYSTEM_INTEGRATION_GUIDE.md) - Detailed architecture
- [IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) - Design rationale
- [web/README.md](web/README.md) - Web app usage
- [M4_COMPLETE.md](docs/M4_COMPLETE.md) - Physics loss details

---

## 💡 Tips

- **Start small**: Use 1-2 scenes with 100 UEs for initial testing
- **Monitor training**: Use Weights & Biases for real-time metrics
- **Validate early**: Check test metrics every 10 epochs
- **Save everything**: Checkpoints are cheap, retraining is expensive
- **Use physics loss**: 10-30% accuracy improvement for NLOS scenarios

---

**Ready to go?** Start with Step 1! 🚀
