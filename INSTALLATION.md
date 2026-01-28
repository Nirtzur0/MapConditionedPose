# Installation Guide

## Understanding the Requirements Files

This project has **three** requirements files with different purposes:

### 1. `requirements.txt` - **Flexible Installation** ⚠️
- Uses **minimum version specifiers** (`>=`)
- **Will install DIFFERENT versions on different computers**
- Example: `torch>=2.0.0` might install 2.8.0 today, 2.9.0 tomorrow
- **Use for:** Development when you want latest compatible versions
- **Risk:** May break compatibility with untested newer versions

### 2. `requirements-pinned.txt` - **Exact Reproducibility** ✅ (Recommended)
- Uses **exact version pins** (`==`)
- **Guarantees identical versions across all installations**
- Contains only the packages actually needed by this project
- **Use for:** Production, reproducible research, sharing with collaborators
- **Benefit:** Everyone gets the exact working configuration

### 3. `requirements-lock.txt` - **Complete Environment Snapshot**
- Full `pip freeze` output with ALL 413 packages
- Includes all transitive dependencies
- **Use for:** Debugging dependency conflicts, archival purposes
- **Note:** Very large, includes development tools and unrelated packages

---

## Recommended Installation Methods

### Method 1: Exact Reproducibility (Recommended) ✅

This ensures you get the **exact same versions** that are currently working:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install lie-learn dependency first (required for escnn)
pip install git+https://github.com/AMLab-Amsterdam/lie_learn.git

# Install exact working versions
pip install -r requirements-pinned.txt

# On macOS with Apple Silicon, you may need for escnn:
LDFLAGS="-L/opt/homebrew/opt/libomp/lib" CPPFLAGS="-I/opt/homebrew/opt/libomp/include" pip install escnn==1.0.11
```

### Method 2: Flexible Installation (Latest Compatible)

This gets the latest versions that satisfy minimum requirements:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install lie-learn dependency first
pip install git+https://github.com/AMLab-Amsterdam/lie_learn.git

# Install with minimum version constraints
pip install -r requirements.txt

# On macOS with Apple Silicon, you may need for escnn:
LDFLAGS="-L/opt/homebrew/opt/libomp/lib" CPPFLAGS="-I/opt/homebrew/opt/libomp/include" pip install escnn
```

⚠️ **Warning:** This may install newer versions that haven't been tested!

### Method 3: Complete Environment Clone

This replicates the **entire environment** including all development tools:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install lie-learn dependency first
pip install git+https://github.com/AMLab-Amsterdam/lie_learn.git

# Install complete frozen environment
pip install -r requirements-lock.txt
```

---

## What's the Difference?

| File | torch version | Result on new computer |
|------|--------------|------------------------|
| `requirements.txt` | `torch>=2.0.0` | Gets **latest** (could be 2.9.0, 3.0.0, etc.) |
| `requirements-pinned.txt` | `torch==2.8.0` | Gets **exactly 2.8.0** |
| `requirements-lock.txt` | `torch==2.8.0` | Gets **exactly 2.8.0** + all 413 deps |

---

## Currently Working Versions (2025-12-30)

These are the **tested and working** versions:

| Package | Version |
|---------|---------|
| Python | 3.x (check with `python --version`) |
| torch | 2.8.0 |
| pytorch-lightning | 2.6.0 |
| tensorflow | 2.20.0 |
| sionna | 1.2.1 |
| numpy | 1.26.4 |
| pandas | 2.3.1 |

See `CURRENT_VERSIONS.md` for complete version list.

---

## Optional Packages

These packages are in `requirements.txt` but **not currently installed** (project works without them):

- `torchvision`, `timm`, `einops` - Vision/ML packages
- `opencv-python` - Image processing (using Pillow instead)
- `laspy`, `pdal`, `plyfile` - LiDAR processing
- `wandb` - Experiment tracking
- `pytest-cov`, `pytest-xdist`, etc. - Testing utilities

Install only if you need these features:
```bash
pip install torchvision timm einops opencv-python laspy[lazrs] pdal plyfile wandb
```

---

## Verification

After installation, verify it works:

```bash
# Quick test
python run_pipeline.py --quick-test

# Full experiment
./run_full_experiment.sh
```

---

## Updating Requirements

If you add new packages and want to update the pinned versions:

```bash
# After installing new packages
pip freeze > requirements-lock.txt

# Then manually update requirements-pinned.txt with only the new packages you added
```

---

## Troubleshooting

### Issue: Different versions installed
**Solution:** Use `requirements-pinned.txt` instead of `requirements.txt`

### Issue: escnn installation fails on macOS
**Solution:** Install lie-learn first, then use LDFLAGS/CPPFLAGS:
```bash
pip install git+https://github.com/AMLab-Amsterdam/lie_learn.git
LDFLAGS="-L/opt/homebrew/opt/libomp/lib" CPPFLAGS="-I/opt/homebrew/opt/libomp/include" pip install escnn==1.0.11
```

### Issue: CUDA/GPU errors
**Solution:** Ensure you have compatible CUDA drivers for PyTorch 2.8.0 and TensorFlow 2.20.0

---

## For Collaborators

**To get the exact working environment:**

1. Clone the repository
2. Use `requirements-pinned.txt` for installation
3. This guarantees you get the same versions that are currently working

**To update to latest versions (at your own risk):**

1. Use `requirements.txt` instead
2. Test thoroughly
3. If it works, update `requirements-pinned.txt` with new versions
