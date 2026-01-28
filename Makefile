.PHONY: help quick-test pipeline train clean setup test

help:
	@echo "Transformer UE Localization - Pipeline Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup          - Install dependencies"
	@echo "  make quick-test     - Run quick end-to-end test (~15 min)"
	@echo "  make pipeline       - Run full pipeline"
	@echo ""
	@echo "Individual Steps:"
	@echo "  make scenes         - Generate 3D scenes only"
	@echo "  make dataset        - Generate dataset only (requires scenes)"
	@echo "  make train          - Train model only (requires dataset)"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run test suite"
	@echo "  make clean          - Remove generated data"
	@echo "  make clean-all      - Remove data + checkpoints"
	@echo ""
	@echo "For more options, see: python run_pipeline.py --help"

setup:
	@echo "Setting up environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"

quick-test:
	@echo "Running quick end-to-end test..."
	. venv/bin/activate && python run_pipeline.py --quick-test

pipeline:
	@echo "Running full pipeline..."
	. venv/bin/activate && python run_pipeline.py --config configs/pipeline.yaml

scenes:
	@echo "Generating scenes..."
	. venv/bin/activate && python scripts/scene_generation/generate_scenes.py \
		--bbox -105.28 40.014 -105.27 40.020 \
		--output data/scenes/boulder_test \
		--num-tx 3 \
		--site-strategy grid

dataset:
	@echo "Generating dataset..."
	. venv/bin/activate && python scripts/generate_dataset.py \
		--scene-dir data/scenes/boulder_test \
		--output-dir data/processed/boulder_dataset \
		--num-trajectories 100 \
		--num-ues 50

dataset-austin:
	@echo "Generating Austin dataset..."
	. venv/bin/activate && python scripts/generate_dataset.py \
		--scene-dir data/scenes/austin_texas \
		--output-dir data/processed/austin_dataset \
		--num-trajectories 100 \
		--num-ues 50

train:
	@echo "Training model..."
	. venv/bin/activate && python scripts/train.py \
		--config configs/training/training_simple.yaml

visualize:
	@echo "Generating visualizations..."
	. venv/bin/activate && python scripts/generate_radio_maps.py \
		--scenes-dir data/scenes/austin_texas \
		--output-dir visualizations \
		--save-plots \
		--plots-dir visualizations/plots \
		--pattern "**/scene_*/scene.xml"

test:
	@echo "Running tests..."
	. venv/bin/activate && pytest tests/ -v

clean:
	@echo "Cleaning generated data..."
	rm -rf data/scenes/boulder_test
	rm -rf data/scenes/quick_test
	rm -rf data/processed/*_dataset
	rm -rf data/synthetic/dataset_*
	@echo "✓ Data cleaned"

clean-all: clean
	@echo "Cleaning checkpoints and logs..."
	rm -rf checkpoints/
	rm -rf lightning_logs/
	rm -f pipeline_*.log
	@echo "✓ All outputs cleaned"

# Convenience targets
run: pipeline
test-quick: quick-test
install: setup
