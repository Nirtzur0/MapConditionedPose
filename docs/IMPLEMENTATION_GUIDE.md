# Implementation Guide: Transformer UE Localization

**Last Updated:** December 23, 2025
**Status:** Planning Phase

---

## Overview

This guide provides a practical roadmap for implementing the Transformer-based UE Localization system. It integrates the **Geo2SigMap v2.0.0** framework for scene generation and leverages the full **Sionna stack (RT+PHY+SYS)** for comprehensive multi-layer feature extraction.

**Strategic Alignment:**
- **Forward Problem (Geo2SigMap):** Environment Signal Map
- **Inverse Problem (Our Project):** Signal Measurements + Map UE Location

---

## PyTorch + Sionna: The Workflow

### Why PyTorch is the Right Choice

**Key Insight:** We use Sionna RT for **data generation only**, not for training. Training happens entirely in PyTorch.

**Framework Separation:**

1. **Data Generation Phase (Sionna RT)**
 - **Sionna RT:** Built on Mitsuba 3 + Dr.Jit, NOT TensorFlow
 - **Sionna PHY/SYS:** TensorFlow-based, used only for feature extraction
 - **Output:** Framework-agnostic formats (NumPy arrays, Zarr, Parquet)
 - **No TensorFlow dependency at training time**

2. **Training Phase (PyTorch)**
 - Load precomputed data from Zarr/Parquet
 - Standard PyTorch DataLoader (simple, fast, familiar)
 - All model code in PyTorch (transformers, encoders, losses)
 - No TensorFlow in training loop

3. **Optional: Physics Loss (Dr.Jit ↔ PyTorch)**
 - Dr.Jit provides `@dr.wrap` for gradient exchange
 - Can query Sionna RT from PyTorch if needed
 - But precomputed maps + bilinear lookup is simpler and faster


## Engineering Milestones

### Milestone 1: Scene Generation Pipeline

**Duration:** 2-3 weeks
**Status:** Not Started

#### Objectives

Establish automated OSM-to-Sionna pipeline with material domain randomization.

#### Deliverables

1. **Scene Builder Script**
 - Input: OSM bounding box or city name
 - Output: Mitsuba XML + `.ply` meshes compatible with Sionna RT
 - Tile size: 512m × 512m (configurable)
 - Features:
 - Building extraction with height attributes
 - Terrain/elevation data (if available)
 - Road network extraction
 - Automated site placement (configurable patterns)

2. **Material Domain Randomization**
 - Extend Geo2SigMap's Mitsuba XML recording
 - Randomize per building/material class:
 - Complex permittivity $\epsilon_r$ (range: 2-15)
 - Conductivity $\sigma$ (range: 0.001-10 S/m)
 - Material class categories: concrete, brick, glass, metal, wood
 - Spatial variation within classes (avoid uniform appearance)

3. **Site Configuration Tool**
 - Programmable site placement (grid, random, custom)
 - Per-site parameters:
 - Position (x, y, z), sector azimuth, downtilt
 - Antenna array config: elements, spacing, pattern
 - Carrier frequency, bandwidth, Tx power
 - Support for multi-site scenarios (3-8 sites per tile)

4. **Validation Suite**
 - Unit tests: coordinate system consistency
 - Sanity checks: building heights, site placement, LoS feasibility
 - Visualization: 3D scene preview in Blender or Mitsuba viewer

#### Technical Stack

- **GIS:** `osmnx`, `geopandas`, `shapely`, `pyproj`, `rasterio`
- **Mesh:** `open3d`, `pyvista`, `trimesh`
- **Scene:** Blender 3.x + Mitsuba-Blender add-on OR Python OSMscene utility
- **Validation:** Sionna RT `load_scene()` function

#### Reference

- Geo2SigMap v2.0.0 `scene_generation` package
- Sionna RT documentation: Blender + OSM workflows

---

### Milestone 2: Multi-Layer Synthetic Data Generator

**Duration:** 3-4 weeks
**Status:** Not Started

#### Objectives

Generate 1M+ UE samples with comprehensive RT+PHY+SYS features across 5-10 city tiles.

#### Deliverables

1. **Sionna Multi-Layer Feature Extraction (All Computed During Data Gen)**

 **Important:** All three layers are computed during data generation using Sionna (which requires TensorFlow for PHY/SYS). These features are then saved to Zarr/NumPy and loaded as static inputs during PyTorch training. No TensorFlow needed at training time.

 **a) Layer 1: RT (Ray Tracer) - Propagation Features**
 - Extract from Sionna RT `Paths` object (Dr.Jit-based, no TensorFlow):
 - Path Gain (PG): quantized, 1 dB steps
 - Time of Arrival (ToA): per path, with NLoS bias model
 - Angle of Arrival (AoA): $(\theta, \phi)$ per path
 - Angle of Departure (AoD): $(\theta^{\text{Tx}}, \phi^{\text{Tx}})$ per path
 - Doppler Shifts: $f_D$ from path velocities
 - RMS Delay Spread: $\tau_{\text{RMS}}$ from PDP

 **b) Layer 2: PHY (Link-Level) - Channel Features**
 - Generate using Sionna PHY layer (TensorFlow-based, runs during data gen only):
 - Channel Impulse Response (CIR): compressed summaries (tap magnitudes, delays)
 - Signal-to-Noise Ratio (SNR): per antenna element
 - Signal-to-Interference-plus-Noise Ratio (SINR): realistic interference
 - Channel State Information (CSI): rank, condition number, compressed
 - Received Signal Strength Indicator (RSSI): with OFDM effects
 - Include OFDM parameter variations:
 - Subcarrier spacing: 15, 30, 60, 120 kHz (5G NR)
 - Cyclic prefix lengths
 - **Output:** All computed features saved to Zarr as NumPy arrays

 **c) Layer 3: SYS (System-Level) - Network Features**
 - Simulate using Sionna SYS layer (TensorFlow-based, runs during data gen only):
 - Throughput estimates: using physical-layer abstraction
 - Block Error Rate (BLER): predictions
 - Handover metrics: serving cell stability indicators
 - Cell IDs: serving + top-K neighbors (K=8)
 - RSRP/RSRQ ratios: per neighbor cell
 - Timing Advance (TA): quantized values
 - **Output:** All computed features saved to Zarr as NumPy arrays

2. **UE Sampling Strategy**
 - Use Sionna's `sample_positions()` with constraints:
 - Minimum SINR/RSS thresholds (realistic coverage)
 - Distance bounds: min 10m, max 1000m from serving site
 - Exclude indoor positions (use building footprint masks)
 - Non-uniform sampling: concentrate on coverage-realistic areas
 - Target: 100K-200K samples per city tile

3. **Measurement Realism**
 - **Quantization:**
 - Power: 1 dB steps (RSRP, RSRQ)
 - TA: 16 Ts units (5G NR)
 - Angle: 1-degree bins
 - **Noise Models:**
 - Shadow fading: log-normal, $\sigma_{\text{SF}}$ = 6-10 dB
 - Thermal noise: realistic SNR floors
 - NLoS bias: ToA delays, $\Delta\tau$ ~ Exp($\lambda$)
 - **Dropout:**
 - Neighbor list truncation: top-K=8 cells only
 - Feature dropout: 10-30% missing features (AoA, TA)
 - Spatial dropout: not all cells report at each time

4. **Temporal Sequence Generation**
 - Generate short time windows: 5-20 reports per UE
 - Irregular timestamps: realistic measurement rates (100-500 ms intervals)
 - Correlated measurements: shadow fading with spatial correlation
 - Motion models: stationary, pedestrian (1 m/s), vehicular (10 m/s)

5. **Dataset Structure**

 **Hierarchical Schema:**
 ```
 Dataset/
 ├── scenes/
 │ ├── city_A_tile_01/
 │ │ ├── scene.xml (Mitsuba)
 │ │ ├── scene_metadata.json (sites, materials, bbox)
 │ │ └── radio_maps/ (precomputed Sionna maps)
 │ └── city_B_tile_01/
 │ └── ...
 ├── samples/
 │ ├── city_A_tile_01.zarr (or .parquet)
 │ │ ├── positions (x, y, z)
 │ │ ├── rt_features (PG, ToA, AoA, AoD, Doppler, RMS-DS)
 │ │ ├── phy_features (CIR, SNR, SINR, CSI, RSSI)
 │ │ ├── sys_features (Throughput, BLER, CellIDs, RSRP/RSRQ, TA)
 │ │ ├── timestamps
 │ │ └── masks (missing token/feature indicators)
 │ └── ...
 └── maps/
 ├── osm_maps/ (building heights, materials, roads)
 └── sionna_radio_maps/ (PG, ToA, SNR, SINR, Throughput)
 ```

 **Storage Format:**
 - Zarr (recommended): chunked, cloud-friendly, efficient random access
 - Alternative: Parquet for tabular features

6. **Precomputed Sionna Radio Maps**
 - Generate using Sionna's `RadioMapSolver`:
 - Resolution: 1-2m per pixel
 - Coverage: full scene (512m × 512m)
 - Features: PG, ToA, AoA, SNR, SINR, Throughput, BLER
 - Per BS configuration: all sites, all frequency bands
 - Store as multi-channel rasters (NumPy arrays or Zarr)
 - Co-register with OSM maps (same spatial reference, resolution)

7. **Data Validation**
 - Sanity checks:
 - Path loss models match free-space at short distances
 - ToA consistent with geometric distances
 - SINR distributions realistic (no anomalies)
 - Visualization:
 - Radio maps (heatmaps)
 - Sample distributions (scatter plots)
 - Feature histograms

#### Technical Stack

- **Simulation (Data Gen Only):**
 - `sionna-rt` (v0.14+) for RT layer features (Dr.Jit-based)
 - `sionna` (v0.18+) for PHY/SYS layer features (TensorFlow-based)
 - `tensorflow` (2.14-2.19) - only needed for PHY/SYS feature computation
 - `mitsuba` (3.0+), `drjit` (0.4+)
- **Data Storage:** `zarr`, `parquet`, `pyarrow`, `h5py`
- **Processing:** `numpy`, `scipy`, `pandas`
- **Validation:** `matplotlib`, `plotly`

**Note:** TensorFlow is ONLY needed during data generation to compute PHY/SYS features. Once features are saved to Zarr, training uses PyTorch exclusively.

#### Reference

- Geo2SigMap ray tracing module (extended)
- Sionna RT: `PathSolver`, `RadioMapSolver`, `sample_positions()`

---

### Milestone 3: Map-Conditioned Transformer (Sionna Training Interface)

**Duration:** 4-5 weeks
**Status:** Not Started

#### Objectives

Implement end-to-end positioning network with dual map conditioning and coarse-to-fine output.

#### Deliverables

1. **Radio Encoder (Temporal Set Transformer)**

 **Input:** Sparse temporal measurement sequence
 - Tokens: `[cell_id, beam_id, time, rt_features, phy_features, sys_features]`
 - Variable length: 1-20 tokens per sequence
 - Missing tokens: masked attention

 **Architecture:**
 - Embedding layers: cell ID, beam ID, time
 - Feature projection: linear layer for RT+PHY+SYS features
 - Transformer encoder: 6-12 layers, 8 heads, 512 hidden dim
 - Masking: attention mask for missing tokens
 - Output: CLS token embedding `z_radio` (512-dim)

2. **Dual Map Encoder**

 **Stream 1: Sionna Radio Maps**
 - Multi-channel input: [PG, ToA, SNR, SINR, Throughput] (5 channels)
 - Resolution: 512×512 at 1m/pixel (coarse stage)

 **Stream 2: OSM Building Maps**
 - Multi-channel input: [height, materials, footprints, roads] (4 channels)
 - Resolution: 512×512 at 1m/pixel (coarse stage)

 **Architecture Options:**
 - **Option A (Recommended):** Early fusion single ViT encoder
 - Concatenate channels: 9 total
 - Patch size: 16×16 32×32 patches
 - ViT: 12 layers, 8 heads, 768 hidden dim
 - Output: grid of 1024 spatial tokens

 - **Option B:** Separate encoders cross-attention
 - Two ViT encoders (one per stream)
 - Cross-attention between streams
 - Concatenate outputs

 **Output:** Spatial token grid `F_maps` (1024 tokens × 768 dim)

3. **Spatio-Temporal Cross-Attention Fusion**

 **Mechanism:**
 - Query: `z_radio` from radio encoder (1 × 512)
 - Keys/Values: `F_maps` from map encoder (1024 × 768)
 - Multi-head cross-attention: 8 heads
 - Output: fused representation (1 × 768)

 **Coarse Heatmap Head:**
 - Linear projection: (768) (1024) logits
 - Softmax probability over 32×32 grid cells
 - Reshape: (1024) (32, 32) heatmap

4. **Fine Refinement Stage**

 **Input:** Top-K coarse cells (K=5)
 - For each candidate cell, extract high-res patch (64×64 at 0.5m/pixel)
 - Crop from dual maps centered at cell center

 **Architecture:**
 - Smaller ViT encoder: 6 layers, 4 heads, 384 hidden dim
 - Input: concatenated radio maps + OSM maps (9 channels, 64×64)
 - Append `z_radio` as additional context
 - Output: per-cell refinement

 **Refinement Head:**
 - Per cell: predict offset (Δx, Δy) within cell
 - Heteroscedastic: predict (μx, μy, σx, σy) for uncertainty
 - Loss: negative log-likelihood

5. **Training Configuration**

 **Loss Function:**
 ```
 L_total = L_coarse + λ_fine * L_fine + λ_phys * L_phys
 ```

 - `L_coarse`: cross-entropy on grid cell classification
 - `L_fine`: NLL for offset regression (Gaussian)
 - `L_phys`: physics-consistency loss (see M5)
 - Weights: λ_fine=1.0, λ_phys=0.1 (tune)

 **Optimizer:**
 - AdamW: lr=1e-4, weight decay=0.01
 - Warmup: 1000 steps, cosine decay
 - Batch size: 64 (adjust for GPU memory)

 **Regularization:**
 - Dropout: 0.1 in transformer layers
 - Layer normalization
 - Gradient clipping: max norm 1.0

 **Data Augmentation:**
 - Random rotation of maps (±5 degrees)
 - Random scaling (±10%)
 - Feature dropout: randomly zero 10-20% of features
 - Temporal dropout: randomly remove 20-30% of time steps

6. **Training Infrastructure**

 **Framework:** PyTorch Lightning
 - Multi-GPU training (DDP)
 - Automatic checkpointing (top-3 models)
 - Early stopping (patience=10 epochs)
 - Mixed precision (FP16)

 **Experiment Tracking:** Weights & Biases (wandb)
 - Log: loss curves, accuracy metrics, heatmap visualizations
 - Hyperparameter sweeps (lr, λ_phys, architecture variants)

7. **Validation & Analysis**

 **During Training:**
 - Validation every epoch
 - Log: median error, 95th percentile, loss components
 - Visualize: predicted heatmaps, error distributions

 **Post-Training:**
 - Error CDF curves
 - Confusion analysis: where does model fail?
 - Ablations: remove map streams individually
 - Interpretability: attention weights, heatmap evolution

#### Technical Stack

- **ML:** `torch`, `pytorch-lightning`, `timm` (ViT backbones)
- **Training:** `wandb`, mixed precision (`torch.cuda.amp`)
- **Data Loading:** `torch.utils.data.DataLoader`, `zarr` backend
- **Augmentation:** `torchvision.transforms`, custom geometric transforms

#### Reference

- Sionna training interface documentation
- Transformer architectures: ViT, BERT, GPT
- Coarse-to-fine papers: cascaded networks, spatial attention

---

### Milestone 4: Differentiable Physics Regularization

**Duration:** 2-3 weeks
**Status:** Not Started

#### Objectives

Integrate physics-consistency loss using precomputed Sionna radio maps with differentiable lookup.

#### Deliverables

1. **Precomputed Sionna Radio Map Generation**

 **For each scene:**
 - Use Sionna's `RadioMapSolver` with comprehensive feature extraction
 - Resolution: 1m per pixel (fine), 2m per pixel (coarse)
 - Coverage: full scene extent (512m × 512m)

 **Features to precompute:**
 - **RT Layer:** Path Gain (dB), ToA (ns), AoA (degrees)
 - **PHY Layer:** SNR (dB), SINR (dB)
 - **SYS Layer:** Throughput (Mbps), BLER (%)

 **Per configuration:**
 - All BS locations and sectors
 - All frequency bands (if multi-band)
 - Store as multi-channel NumPy arrays or Zarr

 **Storage:**
 - Per scene: ~50-100 MB (7 features × 512×512 × float32)
 - Total for 10 scenes: ~1 GB (manageable)

 **When to use:**
 - High-stakes predictions (emergency services)
 - Low confidence from network (high entropy heatmap)
 - Real-time not critical (adds ~10-50ms per refinement)

6. **Ablation Studies**

 **Experiments:**
 - **No physics loss:** λ_phys=0 (baseline)
 - **Precomputed map loss:** λ_phys=0.1 (fast, proposed)
 - **Full RT loss:** Run real-time ray tracing on 10% of batches (expensive)
 - **Feature ablations:** Remove individual features from L_phys

 **Analysis:**
 - Compare: accuracy (CDF curves), training time, inference time
 - Quantify: benefit vs cost (accuracy gain per compute second)
 - Identify: when does physics loss help most? (NLOS? Sparse measurements?)

7. **Dr.Jit Interop (Advanced, Optional)**

 **Alternative Implementation:**
 - Use Sionna's `@dr.wrap` to exchange gradients with Dr.Jit
 - Enables more sophisticated RT-based losses (beyond bilinear lookup)
 - Example: differentiable path tracing for selected samples

 **When to use:**
 - If precomputed maps insufficient (dynamic scenes)
 - If need gradients w.r.t. scene parameters (material learning)
 - Research mode, not production

#### Technical Stack

- **Core:** `torch`, `torch.nn.functional.grid_sample`
- **Optional:** `drjit`, `sionna-rt` (for advanced Dr.Jit interop)
- **Validation:** Custom metrics, visualization scripts

#### Reference

- Sionna RT: `RadioMapSolver`, differentiable radio maps
- Dr.Jit: `@dr.wrap` interop with PyTorch
- Papers: differentiable rendering, neural radiance fields (for interpolation ideas)

---

### Milestone 5: Visualization Tools (TensorBoard + Streamlit)

**Duration:** 2-3 weeks
**Status:** Complete

#### Objectives

Build an interactive visualization interface using **Streamlit** for rapid development with minimal code and beautiful UI out-of-the-box.

#### Core Features

**Why Streamlit?**
- Pure Python (no JavaScript required)
- Beautiful UI components out-of-the-box
- Minimal code (~500 lines vs 2000+ for React)
- Built-in state management
- Instant updates and hot reload
- Easy deployment (Streamlit Cloud, Docker)

**1. Overview Page**
- System architecture visualization
- Model status and device info
- Scene metadata display
- Quick performance statistics

**2. Metrics Dashboard**
- Error CDF plots with Plotly
- Percentile tables and metrics cards
- Model comparison table
- Success rate visualization

**3. Live Inference Mode**
- Upload JSON measurements or generate demo data
- Real-time inference with progress indicators
- Measurement timeline visualization (Plotly)
- Predicted position heatmap
- Top-K candidate positions with uncertainties

**4. Analysis Tools**
- Feature importance (SHAP-style visualization)
- Error analysis by scenario type
- Ablation study results
- Interactive Plotly charts

#### Technical Stack

**Core Framework:**
```
streamlit==1.29.0 # Main framework
plotly==5.18.0 # Interactive visualizations
pandas>=2.0.0 # Data handling
```

**Optional Enhancements:**
```
folium==0.15.1 # Interactive maps (if needed)
streamlit-folium==0.15.1 # Folium integration
pydeck==0.8.0 # WebGL map layers
```

**Model Inference:**
```
torch>=2.0.0 # PyTorch (already in main project)
zarr>=2.16.0 # Data loading
```

#### Implementation

**Single File App:**
`web/app.py` (~450 lines, focused map explorer)

**Key Features:**
```python
# Page navigation with radio buttons
page = st.radio("Navigation", [
 "🏠 Overview",
 " Metrics Dashboard",
 " Live Inference",
 " Analysis"
])

# Cached model loading
@st.cache_resource
def load_model_cached(model_path):
 model = torch.load(model_path)
 return model

# Interactive visualizations with Plotly
fig = create_error_cdf_plot(errors)
st.plotly_chart(fig, use_container_width=True)

# File uploads for measurements
uploaded_file = st.file_uploader("Upload JSON", type=['json'])
measurements = json.load(uploaded_file)

# Real-time inference with spinner
with st.spinner("Running inference..."):
 prediction = model(measurements)
st.success(" Complete!")
```

#### Deployment Options

**1. Local Development (Fastest)**
```bash
cd web
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**2. Streamlit Cloud (Free, Easiest)**
- Push to GitHub
- Connect at [share.streamlit.io](https://share.streamlit.io)
- One-click deployment
- Automatic updates on git push

**3. Docker (Production)**
```bash
docker build -t ue-localization-web web/
docker run -p 8501:8501 ue-localization-web
```

**4. Custom Server**
```bash
streamlit run streamlit_app.py \
 --server.port 8080 \
 --server.address 0.0.0.0
```

#### Deliverables

1. **Streamlit Web App** - Single-file implementation (~500 lines)
2. **Interactive Visualizations** - Plotly-based charts and graphs
3. **Live Inference** - Upload measurements and get predictions
4. **Metrics Dashboard** - Comprehensive performance analysis
5. **Deployment Config** - Dockerfile and Streamlit config
6. **Documentation** - Complete README with usage examples

**Files Created:**
```
web/
├── app.py # Focused map explorer (~450 lines)
├── requirements.txt # Python dependencies
├── README.md # Usage documentation
├── Dockerfile # Container deployment
└── .streamlit/
 └── config.toml # Theme and settings
```

#### Success Metrics

 **Development Speed:** 10× faster than React+FastAPI approach
 **Code Maintainability:** Single Python file, no JavaScript
 **Deployment Simplicity:** One command to run locally or deploy
 **Inference Latency:** <100ms (backend model) + <50ms (Streamlit render)
 **User Experience:** Beautiful default UI, responsive design
 **Browser Support:** All modern browsers via Streamlit

#### Comparison: Streamlit vs Full-Stack

| Aspect | Streamlit | React + FastAPI |
|--------|-----------|-----------------|
| **Lines of Code** | ~500 | ~2000+ |
| **Languages** | Python only | Python + JavaScript/TypeScript |
| **Setup Time** | 5 minutes | 2-3 hours |
| **Deployment** | 1 command | Docker Compose + nginx |
| **Maintenance** | Single file | Multiple services |
| **UI Quality** | Beautiful default | Requires styling |
| **Learning Curve** | Minimal | React + API design |
| **Cost** | Free (Streamlit Cloud) | Server costs |

**Recommendation:** Streamlit is ideal for research/demo interfaces. For production at scale (100+ concurrent users), consider full-stack approach.

---

## Development Timeline

### Phase 1: Infrastructure (Weeks 1-3)
- Environment setup (Sionna, PyTorch, GIS libs)
- M1: Scene generation pipeline
- Basic unit tests and validation

### Phase 2: Data Generation (Weeks 4-7)
- M2: Multi-layer synthetic data generator
- Generate 1M+ samples across 5-10 city tiles
- Precompute Sionna radio maps and OSM maps

### Phase 3: Core Model (Weeks 8-12)
- M3: Implement map-conditioned transformer
- Training infrastructure (Lightning, wandb)
- Initial training runs and ablations

### Phase 4: Physics Integration (Weeks 13-15)
- M4: Physics-consistency loss implementation
- Ablation studies
- Inference-time refinement experiments

### Phase 5: Validation & Analysis (Weeks 16-18)
- Test on unseen cities
- Comprehensive evaluation
- Paper/report writing

### Phase 6: Web Interface (Weeks 19-21)
- M5: Streamlit-based visualization interface
- Interactive dashboards (Plotly visualizations)
- Live inference mode with file upload
- Deployment configuration (Docker + Streamlit Cloud)


## Validation Strategy

### Generalization Testing

**Unseen Cities (Critical):**
- Train on cities: A, B, C
- Validate on cities: D, E (from same regions)
- Test on cities: F, G (different regions/morphologies)

**Urban Morphology Diversity:**
- Dense urban (Manhattan, Hong Kong)
- Suburban (low-rise, residential)
- Mixed (downtown + residential)
- Campus (large open spaces + buildings)

### Robustness Testing

**Measurement Sparsity:**
- K=3, 5, 8 neighbor cells
- 1, 5, 10, 20 temporal reports
- Feature dropout: 0%, 20%, 50%

**Map Quality:**
- Perfect maps (training condition)
- Outdated maps (buildings changed)
- Incorrect maps (material errors)
- Missing maps (OSM-only or Radio-only)

**Environmental Conditions:**
- Different frequencies (3.5 GHz, 28 GHz)
- Different site densities (sparse, dense)
- Different materials (glass-heavy, concrete-heavy)

---

## Success Criteria

### Accuracy Targets

**Vs. Strongest Cell Centroid:**
- 50th percentile: <50m (vs ~100m)
- 95th percentile: <150m (vs ~500m)

**Vs. KNN Fingerprinting:**
- 50th percentile: <20m (vs ~50m)
- 95th percentile: <80m (vs ~200m)

**Absolute:**
- 50th percentile: <15m (good)
- 67th percentile: <30m (acceptable)
- 95th percentile: <100m (edge cases)

### Generalization Metrics

**Unseen Cities:**
- Accuracy degradation <20% vs seen cities
- No catastrophic failures (>500m errors <5%)

**Sparse Measurements:**
- K=3 cells: degradation <30% vs K=8
- 1 time step: degradation <50% vs 20 steps

### Computational Efficiency

**Inference Time:**
- <100ms per prediction on GPU
- <500ms on CPU (optional)

**Model Size:**
- <100M parameters (target)
- <200M parameters (acceptable)

---

## Risk Mitigation

### Technical Risks

**1. Sim-to-Real Gap**
- **Risk:** Model trained on simulation doesn't work on real data
- **Mitigation:** Aggressive domain randomization, plan for fine-tuning on real data
- **Fallback:** Hybrid approach (combine with classical methods)

**2. Map Dependency**
- **Risk:** Model fails when maps are outdated/incorrect
- **Mitigation:** Test with deliberately corrupted maps, train with map augmentation
- **Fallback:** Graceful degradation (reduce confidence, fall back to radio-only)

**3. Computational Complexity**
- **Risk:** Model too slow for real-time inference
- **Mitigation:** Optimize architecture, quantization, distillation
- **Fallback:** Coarse-only mode (skip fine refinement)

**4. Data Generation Bottleneck**
- **Risk:** Ray tracing too slow to generate sufficient data
- **Mitigation:** Parallelization, cloud compute, precompute maps aggressively
- **Fallback:** Reduce dataset size, focus on quality over quantity

**5. Framework Interoperability**
- **Risk:** Complications from mixing Sionna (TensorFlow) and PyTorch
- **Reality:** Not actually a risk - data generation and training are fully separated
- **Workflow:** Generate data with Sionna Save to Zarr Train with PyTorch (no mixing)
- **Optional:** Dr.Jit `@dr.wrap` enables PyTorch ↔ Sionna RT if needed (advanced feature)

### Research Risks

**1. Physics Loss Not Helpful**
- **Risk:** Physics loss doesn't improve accuracy vs standard supervised
- **Mitigation:** Ablation studies, tune λ_phys carefully
- **Fallback:** Remove physics loss, focus on architecture/data

**2. Transformer Overkill**
- **Risk:** Simpler models (CNN, MLP) achieve similar accuracy
- **Mitigation:** Rigorous baseline comparisons, justify complexity
- **Fallback:** Use simpler model if transformer not needed

**3. Map Encoding Ineffective**
- **Risk:** Model doesn't learn to use map information
- **Mitigation:** Explicit map-measurement alignment losses, interpretability analysis
- **Fallback:** Handcrafted map features (Option C from architecture)

---

## Future Enhancements

### Near-Term (6-12 months)

1. **Real-World Validation**
 - Collect small real measurement dataset (100-1000 samples)
 - Fine-tune on real data
 - Quantify sim-to-real gap

2. **Multi-Band Support**
 - Train on mixed frequencies (sub-6, mmWave)
 - Study frequency-dependent performance

3. **Tracking Extension**
 - Temporal filtering (Kalman, particle filter)
 - Motion models (pedestrian, vehicular)

### Long-Term (1-2 years)

1. **Online Learning**
 - Adapt to site-specific characteristics
 - Learn from measurement feedback

2. **Multi-Modal Fusion**
 - Integrate GNSS (when available)
 - IMU for motion priors
 - Visual/LiDAR for geometric context

3. **System-Level Integration**
 - Joint positioning and handover
 - Network optimization (positioning + resource allocation)
 - Multi-user collaborative positioning

---

## Conclusion

This implementation guide provides a practical roadmap for building a physics-informed, map-conditioned UE localization system. By leveraging Geo2SigMap for scene generation and Sionna's full stack (RT+PHY+SYS) for multi-layer feature extraction, we can train a transformer-based model that combines:

1. **Temporal measurement sequences** (sparse, realistic)
2. **Physics-based radio maps** (precomputed Sionna)
3. **Geometric building maps** (OSM)

The coarse-to-fine architecture with differentiable physics regularization enables accurate positioning while maintaining computational efficiency. Rigorous validation on unseen cities and diverse scenarios ensures the approach generalizes beyond training conditions.

**Next Steps:** Begin with M1 (scene generation) and establish the data pipeline foundation.
