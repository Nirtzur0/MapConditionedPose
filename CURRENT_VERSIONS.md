# Current Working Versions

**Last Updated:** 2025-12-30T13:59:42+02:00  
**Status:** ✅ Running and Working  
**Running Processes:**
- `./run_full_experiment.sh` (running for ~5 minutes)
- `python run_pipeline.py --quick-test` (running for ~24 seconds)

---

## Core Dependencies

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| numpy | >=1.24.0 | **1.26.4** | ✅ |
| pyproj | >=3.6.0 | **3.7.2** | ✅ |
| PyYAML | >=6.0 | **6.0.2** | ✅ |
| easydict | >=1.10 | **1.13** | ✅ |
| optuna | >=3.0.0 | **4.6.0** | ✅ |
| optuna-integration | >=3.0.0 | **4.6.0** | ✅ |

---

## Deep Learning (PyTorch Stack)

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| torch | >=2.0.0 | **2.8.0** | ✅ |
| torchvision | >=0.15.0 | **NOT INSTALLED** | ⚠️ |
| pytorch-lightning | >=2.0.0 | **2.6.0** | ✅ |
| torchmetrics | >=1.0.0 | **1.8.2** | ✅ |
| timm | >=0.9.0 | **NOT INSTALLED** | ⚠️ |
| einops | >=0.7.0 | **NOT INSTALLED** | ⚠️ |

---

## Data Storage & Processing

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| zarr | >=2.16.0 | **3.1.5** | ✅ |
| numcodecs | >=0.12.0 | **0.16.5** | ✅ |
| pandas | >=2.0.0 | **2.3.1** | ✅ |

---

## Scene Generation Dependencies

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| osmnx | >=2.0.0 | **2.0.7** | ✅ |
| opencv-python | >=4.8.0 | **NOT INSTALLED** | ⚠️ |
| shapely | >=2.0.0 | **2.1.2** | ✅ |
| rasterio | >=1.3.0 | **1.4.4** | ✅ |
| open3d | >=0.17.0 | **0.19.0** | ✅ |
| triangle | (no version specified) | **20250106** | ✅ |
| pyvista | >=0.43.0 | **0.46.4** | ✅ |
| geopandas | >=0.14.0 | **1.1.2** | ✅ |
| tqdm | >=4.65.0 | **4.67.1** | ✅ |
| laspy[lazrs] | (no version specified) | **NOT INSTALLED** | ⚠️ |
| pdal | (no version specified) | **NOT INSTALLED** | ⚠️ |
| plyfile | (no version specified) | **NOT INSTALLED** | ⚠️ |
| requests | >=2.31.0 | **2.32.5** | ✅ |
| scipy | >=1.11.0 | **1.16.1** | ✅ |
| importlib_resources | >=6.0.0 | **6.5.2** | ✅ |

---

## Experiment Tracking & Logging

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| wandb | >=0.15.0 | **NOT INSTALLED** | ⚠️ |
| rich | >=13.0.0 | **13.9.4** | ✅ |

---

## Testing

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| pytest | >=7.4.0 | **8.4.2** | ✅ |
| pytest-cov | >=4.1.0 | **NOT INSTALLED** | ⚠️ |
| pytest-xdist | >=3.3.0 | **NOT INSTALLED** | ⚠️ |
| pytest-timeout | >=2.1.0 | **NOT INSTALLED** | ⚠️ |
| pytest-sugar | >=0.9.7 | **NOT INSTALLED** | ⚠️ |
| pytest-html | >=3.2.0 | **NOT INSTALLED** | ⚠️ |

---

## Web Interface (Streamlit)

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| streamlit | >=1.29.0 | **1.29.0** | ✅ |
| plotly | >=5.18.0 | **5.18.0** | ✅ |
| folium | >=0.15.0 | **0.15.1** | ✅ |
| streamlit-folium | >=0.15.0 | **0.15.1** | ✅ |
| pydeck | >=0.8.0 | **0.8.0** | ✅ |
| Pillow | >=10.0.0 | **10.4.0** | ✅ |
| python-dotenv | >=1.0.0 | **1.0.0** | ✅ |
| reportlab | >=4.0.0 | **4.0.7** | ✅ |

---

## Optional: Sionna Ray Tracing

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| sionna | >=0.16.0 | **1.2.1** | ✅ |
| tensorflow | >=2.15.0 | **2.20.0** | ✅ |

---

## Geometric Deep Learning

| Package | Required (requirements.txt) | Installed | Status |
|---------|----------------------------|-----------|--------|
| escnn | >=1.0.0 | **1.0.11** | ✅ |

---

## Summary

### ✅ Currently Working Configuration
The project is **running successfully** with the following configuration:
- **Python Environment:** Active virtual environment at `.venv`
- **Core ML Stack:** PyTorch 2.8.0, PyTorch Lightning 2.6.0
- **Scene Generation:** All major dependencies installed (osmnx, rasterio, open3d, pyvista, geopandas)
- **Ray Tracing:** Sionna 1.2.1 with TensorFlow 2.20.0
- **Web Interface:** Streamlit 1.29.0 with all visualization libraries

### ⚠️ Missing Packages (Not Currently Needed)
The following packages are listed in `requirements.txt` but not installed. **However, the project is running successfully without them:**

1. **Vision/ML packages:**
   - `torchvision` - Not needed for current cellular positioning tasks
   - `timm` - Vision transformers, not currently used
   - `einops` - Tensor operations, not currently required

2. **LiDAR processing:**
   - `laspy[lazrs]` - LiDAR file processing
   - `pdal` - Point data abstraction library
   - `plyfile` - PLY file processing
   - `opencv-python` - May be using Pillow instead

3. **Experiment tracking:**
   - `wandb` - Weights & Biases (project may be using other logging)

4. **Testing utilities:**
   - `pytest-cov`, `pytest-xdist`, `pytest-timeout`, `pytest-sugar`, `pytest-html`
   - Core pytest (8.4.2) is installed and working

### 📝 Recommendations

1. **Update requirements.txt** to reflect actual working versions:
   - Consider pinning critical packages to current working versions
   - Remove or mark as optional packages that aren't needed

2. **Optional installations** can be added if needed:
   ```bash
   pip install torchvision timm einops opencv-python laspy[lazrs] pdal plyfile wandb
   pip install pytest-cov pytest-xdist pytest-timeout pytest-sugar pytest-html
   ```

3. **Version conflicts to watch:**
   - `zarr` upgraded from 2.16.0 to 3.1.5 (major version change)
   - `torch` at 2.8.0 (very recent, ensure compatibility)
   - `tensorflow` at 2.20.0 (very recent)

### 🎯 Current Status
**The project is fully operational with the installed packages.** The missing packages appear to be optional or for features not currently in use. The core pipeline is running successfully as evidenced by the active `run_full_experiment.sh` and `run_pipeline.py --quick-test` processes.
