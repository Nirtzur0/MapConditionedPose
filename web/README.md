# UE Positioning Model Explorer

Simple, focused tool for exploring cellular positioning model predictions on a map.

## Quick Start

```bash
# Start the prediction explorer
streamlit run web/app.py --server.port 8501
```

## Services

### 📊 Comet ML (https://www.comet.com/)
**For Training Monitoring**
- Real-time loss curves and metrics
- Learning rate schedules and hyperparameters
- Model computational graph
- Experiment comparison and tracking
- Set COMET_API_KEY to enable

### 📍 Streamlit App (http://localhost:8501)  
**For Prediction Exploration**
- Interactive map showing ground truth vs predictions
- Error visualization with color-coded markers
- Uncertainty ellipses (if model provides them)
- Error distribution histograms and CDFs
- Per-sample error analysis

## What You'll See

### Main Map View
- **Green circles**: Ground truth UE positions
- **Red X markers**: Model predictions (color intensity = error magnitude)
- **Red lines**: Connect GT to prediction showing error vector
- **Orange ellipses**: Uncertainty regions (1-sigma)

### Error Analysis
- Histogram of positioning errors
- Cumulative distribution function (CDF)
- Key percentiles: P50, P90, P95
- Success rates at different thresholds

## Features

- Load any dataset from `data/processed/quick_test_dataset/`
- Automatically loads best trained model from `checkpoints/best_model.pt`
- Adjustable number of samples to visualize
- Toggle uncertainty visualization
- Real-time prediction generation from trained model

## Philosophy

The old app had too many placeholder pages with fake data. This new version focuses on **one thing**: 
showing you where your model thinks the UE is versus where it actually is, with all the visual intuition you need.

For training metrics, use TensorBoard - that's what PyTorch Lightning is designed for.
