# 📡 UE Localization - Milestone 5 Complete! ✅

## Visualization Tools

Focused, practical tools for training monitoring and prediction exploration.

### 🎉 What's Implemented

✅ **TensorBoard Integration** (PyTorch Lightning built-in)
- Real-time loss curves (total, coarse, fine)
- Learning rate monitoring  
- Gradient histograms
- Model computational graph
- Hyperparameter tracking
- All metrics logged automatically

✅ **Streamlit Map Explorer** (~450 lines of Python)
- Interactive map showing GT vs Predictions
- Error visualization with color-coded markers
- Uncertainty ellipses
- Error distribution analysis (histogram, CDF)
- Percentile metrics (P50, P90, P95)

### Quick Start

```bash
# Start both monitoring tools
./start_monitoring.sh

# Or individually:
tensorboard --logdir lightning_logs --port 6006
streamlit run web/app.py --server.port 8501
```

- **TensorBoard**: `http://localhost:6006` (training metrics)
- **Streamlit**: `http://localhost:8501` (prediction explorer)

### File Structure

```
web/
├── app.py                      # Focused map explorer (~450 lines) ✅
├── demo_measurements.json      # Sample data
├── requirements.txt            # Dependencies
└── README.md                   # Usage docs

start_monitoring.sh             # Launch both tools ✅
scripts/train.py                # Training with TensorBoard enabled ✅
```

### Features

#### 1. Map Visualization (Streamlit)
- **Green circles**: Ground truth UE positions
- **Red X markers**: Model predictions (color = error magnitude)
- **Red lines**: Error vectors connecting GT to prediction
- **Orange ellipses**: Uncertainty regions (1-sigma)
- Interactive zoom/pan with Plotly

#### 2. TensorBoard (Training Monitoring)
- Real-time loss curves (total, coarse, fine)
- Learning rate schedules
- Gradient histograms and norms
- Model computational graph
- Hyperparameter tracking

#### 3. Error Analysis (Streamlit)
- Error histograms
- Cumulative distribution function (CDF)
- Key percentiles: P50, P90, P95
- Success rates at different thresholds
- Per-sample inspection

### Design Philosophy

**"Do one thing and do it well"**

- Use PyTorch's built-in TensorBoard for training metrics
- Use custom Streamlit app for spatial prediction exploration
- No redundant pages with fake data
- Focus on maximum intuition from visualization

### Example Usage

**Training with TensorBoard:**
```bash
python scripts/train.py --config configs/training_simple.yaml

# In another terminal:
tensorboard --logdir lightning_logs
```

**Exploring Predictions:**
```bash
streamlit run web/app.py

# Select dataset from sidebar
# Toggle predictions/uncertainty
# Adjust number of samples
# View interactive map
```

### Technical Details

**Data Loading:**
- Loads zarr datasets from `data/processed/quick_test_dataset/`
- Reads ground truth positions (ue_x, ue_y)
- Extracts RT measurements for model input

**Model Inference:**
- Loads trained checkpoint from `checkpoints/best_model.pt`
- Runs real predictions on selected samples
- Extracts position, uncertainty, and heatmaps
- Caches results for fast exploration

**Visualization:**
- Plotly for interactive maps
- Color-coded error intensity
- Uncertainty visualization with ellipses
- Error distribution histograms and CDFs

### Dependencies

```bash
pip install tensorboard streamlit plotly numpy pandas torch pytorch-lightning zarr
```

### Status

- ✅ TensorBoard logging enabled in training
- ✅ Focused Streamlit map explorer implemented
- ✅ Real model inference on real data
- ✅ Interactive error visualization
- ✅ Uncertainty visualization
- ✅ Launch script for both tools
- ✅ Documentation updated

### Archived

- `web/streamlit_app.py` - Old 844-line multi-page app (removed)
- Replaced with focused 450-line map explorer

