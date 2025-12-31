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

# Experiment Settings
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STUDY_NAME="ue-localization-${TIMESTAMP}"
N_TRIALS=10
RUN_NAME="multi_city_optuna_${TIMESTAMP}"

# Datasets
# Define explicit Evaluation Dataset (Strictly held-out)
EVAL_DATASET="data/processed/sionna_dataset_eval/dataset_eval.zarr"

# --- Execution ---
echo "🚀 Starting Full Multi-City Optuna Pipeline"
echo "   Study: $STUDY_NAME"
echo "   Trials: $N_TRIALS"
echo "   Run Name: $RUN_NAME"
echo "   Test Set: $EVAL_DATASET"

# Activate virtual environment
source .venv/bin/activate

# Run the pipeline
# NOTE: 
# --train-datasets is NOT specified, so the pipeline will 'discover' and use ALL other scenes found in data/scenes/
# --eval-dataset is specified, so this dataset will be used STRICTLY for Testing (post-training evaluation).
# Optuna will optimize based on a 20% validation split of the Training data, keeping the Test set unseen.

python run_pipeline.py \
  --scene-config configs/scene_generation/scene_generation.yaml \
  --data-config configs/data_generation/data_generation_sionna.yaml \
  --optimize \
  --n-trials "$N_TRIALS" \
  --study-name "$STUDY_NAME" \
  --eval-dataset "$EVAL_DATASET" \
  --run-name "$RUN_NAME" \
  --clean \
  "$@"

echo "✅ Pipeline successfully finished!"
