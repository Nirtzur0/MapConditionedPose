# UE Localization System Explorer

Streamlit app aligned with the current pipeline outputs under `outputs/<experiment>/`.
It is meant to *inspect experiments* (config + report), *browse datasets*, and *run model predictions* from actual checkpoints.

## Quick Start

```bash
streamlit run web/app.py --server.port 8501
```

If your outputs live elsewhere:

```bash
MCP_OUTPUTS_DIR=/path/to/outputs streamlit run web/app.py --server.port 8501
```

## What It Does Now

- **Experiment Overview**
  - Reads `outputs/<experiment>/config.yaml`
  - Reads `outputs/<experiment>/report.yaml`

- **Dataset Explorer**
  - Lists `outputs/<experiment>/data/*.lmdb`
  - Shows ground-truth point clouds and range stats
  - Supports split selection (`all/train/val/test`)

- **Predictions**
  - Lists `outputs/<experiment>/checkpoints/*.ckpt`
  - Runs inference on a random subset
  - Shows GT vs prediction map with error vectors and summary metrics

- **Sample Viewer**
  - Inspect a single sample (scene_id, scene_idx, position)
  - View radio map + OSM map channels
  - Quick measurement summary statistics

## Expected Output Layout

```
outputs/
  <experiment>/
    config.yaml
    report.yaml
    data/
      dataset_*.lmdb
    checkpoints/
      *.ckpt
```

## Notes

- If you only ran data generation, the Predictions tab will remain disabled.
- If you generated split datasets (train/val/test LMDBs), you can still use `split=all`.
- Heavy datasets are cached with Streamlit; restart the app if you regenerate LMDBs.
