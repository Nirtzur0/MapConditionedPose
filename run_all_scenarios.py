#!/usr/bin/env python3
"""
Run all data generation scenarios.
"""

import subprocess
import sys
from pathlib import Path

def run_scenario(scenario_name, scene_config, data_config):
    """Runs the pipeline for a single scenario."""
    print(f"--- Running scenario: {scenario_name} ---")
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--scene-config",
        str(scene_config),
        "--data-config",
        str(data_config),
        "--clean", # Clean previous runs
        "--run-name",
        f"pipeline_{scenario_name}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"--- Scenario {scenario_name} completed successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Scenario {scenario_name} failed with exit code {e.returncode} ---")
        # Decide if you want to stop on failure or continue
        # sys.exit(e.returncode)

def main():
    """Main function to run all scenarios."""
    project_root = Path(__file__).parent
    configs_dir = project_root / "configs"
    
    scenarios = {
        "paris": {
            "scene_config": configs_dir / "scene_generation" / "scene_generation_paris.yaml",
            "data_config": configs_dir / "data_generation" / "data_generation_paris.yaml",
        },
        "nyc": {
            "scene_config": configs_dir / "scene_generation" / "scene_generation_nyc.yaml",
            "data_config": configs_dir / "data_generation" / "data_generation_nyc.yaml",
        },
        "tokyo": {
            "scene_config": configs_dir / "scene_generation" / "scene_generation_tokyo.yaml",
            "data_config": configs_dir / "data_generation" / "data_generation_tokyo.yaml",
        },
    }
    
    for name, configs in scenarios.items():
        run_scenario(name, configs["scene_config"], configs["data_config"])
        
    print("--- All scenarios completed ---")

if __name__ == "__main__":
    main()
