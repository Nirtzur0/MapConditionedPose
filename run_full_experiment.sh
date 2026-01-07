#!/bin/bash

# Exit on error
set -e

# --- Configuration ---
# Comet ML Credentials
export COMET_API_KEY="1lc3SG8vCkNzrn5p9ZmZs328K"
export COMET_WORKSPACE="nirtzur0"
export COMET_PROJECT_NAME="ue-localization"
export OVERPASS_URL="https://overpass.kumi.systems/api/interpreter"

# Suppress noisy TensorFlow/cuFFT registration warnings (caused by conflict with PyTorch)
export TF_CPP_MIN_LOG_LEVEL=3

# --- Script Options ---
# Set USE_OPTUNA=false to run single training with optimized config instead of hyperparameter search
USE_OPTUNA="${USE_OPTUNA:-true}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training/training_optimized.yaml}"

# Experiment Settings
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STUDY_NAME="ue-localization-${TIMESTAMP}"
N_TRIALS=10

# Datasets
# Training dataset (70k samples from 7 cities)
TRAIN_DATASET="data/processed/sionna_dataset/dataset_20260107_080756.zarr"
# Define explicit Evaluation Dataset (Strictly held-out) - optional
EVAL_DATASET="data/processed/sionna_dataset_eval/dataset_eval.zarr"

# --- Dynamic Configuration Based on Mode ---
if [ "$USE_OPTUNA" = "true" ]; then
    RUN_NAME="multi_city_optuna_${TIMESTAMP}"
    STORAGE="sqlite:///optuna_studies.db"
    MODE_DESC="Optuna Hyperparameter Search"
else
    RUN_NAME="single_run_${TIMESTAMP}"
    MODE_DESC="Single Training Run (Optimized Config)"
fi

# --- Execution ---
echo "🚀 Starting Pipeline: $MODE_DESC"
echo "   Run Name: $RUN_NAME"
echo "   Timestamp: $TIMESTAMP"

if [ "$USE_OPTUNA" = "true" ]; then
    echo "   Study: $STUDY_NAME"
    echo "   Trials: $N_TRIALS"
    echo "   Storage: $STORAGE"
else
    echo "   Config: $TRAINING_CONFIG"
    echo "   Dataset: $TRAIN_DATASET"
fi

if [ -f "$EVAL_DATASET" ]; then
    echo "   Test Set: $EVAL_DATASET"
fi

echo ""

# Activate virtual environment
source .venv/bin/activate

# Run the pipeline
if [ "$USE_OPTUNA" = "true" ]; then
    # Optuna mode: Hyperparameter search
    # NOTE: --train-datasets is NOT specified, so the pipeline will 'discover' and use ALL scenes in data/scenes/
    # Optuna will optimize based on a 20% validation split of the Training data
    python run_pipeline.py \
      --scene-config configs/scene_generation/scene_generation.yaml \
      --data-config configs/data_generation/data_generation_sionna.yaml \
      --optimize \
      --n-trials "$N_TRIALS" \
      --study-name "$STUDY_NAME" \
      --storage "$STORAGE" \
      --eval-dataset "$EVAL_DATASET" \
      --run-name "$RUN_NAME" \
      --clean \
      "$@"
else
    # Single training mode: Use optimized config
    # Skip scene/dataset generation, just train on existing dataset
    if [ ! -f "$TRAIN_DATASET/trajectories/ue_x/.zarray" ]; then
        echo "❌ Error: Training dataset not found at $TRAIN_DATASET"
        echo "   Run data generation first or set TRAIN_DATASET to existing dataset"
        exit 1
    fi
    
    python run_pipeline.py \
      --train-only \
      --config "$TRAINING_CONFIG" \
      --train-datasets "$TRAIN_DATASET" \
      --run-name "$RUN_NAME" \
      "$@"
fi

echo ""
echo "✅ Pipeline successfully finished!"
echo "📊 View results at: https://www.comet.com/$COMET_WORKSPACE/$COMET_PROJECT_NAME"
