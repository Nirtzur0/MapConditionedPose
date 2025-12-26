# Technical Architecture 📐

Physics-informed deep learning for cellular UE localization.

---

## Problem Statement

Given sparse radio measurements $M_t$ from a 5G network, estimate the UE position $\mathbf{x} \in \mathbb{R}^2$ by solving:

$$\hat{\mathbf{x}} = \arg\max_{\mathbf{x}} \, p(\mathbf{x} \mid M_t, R_{\text{maps}}, G_{\text{maps}})$$

Where:
- $M_t$: Temporal measurement sequence (RSRP, ToA, AoA, TA, ...)
- $R_{\text{maps}}$: Precomputed Sionna radio maps (physics priors)
- $G_{\text{maps}}$: OSM building/terrain maps (geometric constraints)

---

## Input Modalities

### 1. Radio Measurements ($M_t$)

Hierarchical features from the 3GPP protocol stack:

| Layer | Features | Source |
|-------|----------|--------|
| **RT** | Path gain, ToA, AoA/AoD, Doppler, RMS-DS | Sionna ray tracing |
| **PHY/FAPI** | RSRP, RSRQ, SINR, CQI, RI, PMI, L1-RSRP/beam | UE PHY reports |
| **MAC/RRC** | TA, cell IDs, neighbor list, throughput, BLER | Protocol stack |

**Characteristics**:
- Variable-length sequences (5-20 reports)
- Irregular timestamps (80-480ms intervals)
- Missing features (dropout, unavailability)
- Quantized values (1 dB RSRP, 16 Ts timing advance)

### 2. Radio Maps ($R_{\text{maps}}$)

Dense spatial grids from Sionna `RadioMapSolver`:

| Channel | Description | Range |
|---------|-------------|-------|
| Path Gain | Received power (dB) | -160 to -40 |
| ToA | Propagation delay (ns) | 0 to 10,000 |
| AoA | Angle of arrival (rad) | -π to π |
| SNR | Signal-to-noise (dB) | -10 to 40 |
| SINR | With interference (dB) | -10 to 30 |
| Throughput | Estimated rate (Mbps) | 0 to 1000 |
| BLER | Block error rate | 0 to 1 |

**Resolution**: 1-2 m/pixel | **Size**: 512×512 typical

### 3. OSM Maps ($G_{\text{maps}}$)

Geometric context from OpenStreetMap:

| Channel | Content |
|---------|---------|
| Height | Building heights (m) |
| Material | Permittivity class |
| Footprint | Building mask |
| Road | Street network |
| Terrain | Ground elevation |

**Purpose**: Enforce geometric feasibility—UEs must be on streets, not inside buildings.

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Radio Encoder                            │
│   [CLS] + Embed(cell,beam,time) + Features(rt,phy,mac)     │
│                           ↓                                 │
│              6-12 Transformer Layers                        │
│                           ↓                                 │
│                    z_radio (CLS)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                Cross-Attention Fusion                       │
│   Q = z_radio  |  K,V = [F_radio_map ; F_osm_map]          │
│                           ↓                                 │
│            Attention(Q, K, V) → z_fused                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Output Heads                             │
│   CoarseHead: p(cell | z_fused) → grid heatmap             │
│   FineHead: (Δx, Δy, σ_x, σ_y) → offset + uncertainty      │
└─────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────┐
│                    Map Encoder                              │
│   Radio Maps → ViT Patches → F_radio_map                   │
│   OSM Maps   → ViT Patches → F_osm_map                     │
└─────────────────────────────────────────────────────────────┘
```

### Attention Mechanism

Cross-attention fuses radio measurements with spatial features:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q = W_Q z_{\text{radio}}$ and $K, V$ are projected from concatenated map features.

### Coarse-to-Fine Output

1. **Coarse**: Classify into grid cells (e.g., 32×32 over 512m scene)
2. **Fine**: Regress sub-cell offset with heteroscedastic uncertainty

This handles multi-modal posteriors (e.g., NLOS ambiguity) naturally.

---

## Loss Functions

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{coarse}} + \lambda_f \mathcal{L}_{\text{fine}} + \lambda_p \mathcal{L}_{\text{phys}}$$

### Coarse Loss (Cross-Entropy)

$$\mathcal{L}_{\text{coarse}} = -\sum_c y_c \log \hat{p}_c$$

Where $y_c = 1$ for the true cell, 0 otherwise.

### Fine Loss (NLL with Uncertainty)

$$\mathcal{L}_{\text{fine}} = \frac{(\mathbf{x} - \hat{\mu})^T \hat{\Sigma}^{-1} (\mathbf{x} - \hat{\mu})}{2} + \frac{\log |\hat{\Sigma}|}{2}$$

This encourages the model to be uncertain when it should be.

### Physics Loss (Differentiable Lookup)

$$\mathcal{L}_{\text{phys}} = \sum_{f \in \mathcal{F}} w_f \left\| m_f^{\text{obs}} - R_f(\hat{\mathbf{x}}) \right\|^2$$

Where $R_f(\hat{\mathbf{x}})$ is bilinear interpolation from the radio map at predicted position.

**Feature weights**: path_gain (1.0), ToA (0.5), AoA (0.3), SNR/SINR (0.8)

**Gradient flow**: Fully differentiable via `F.grid_sample`.

---

## Data Generation Pipeline

```
M1: Scene Generation
    OSM → Geo2SigMap → scene.xml + meshes
         + Material randomization (ITU-R P.2040)
         + Site placement (grid/random/ISD)
                    ↓
M2: Data Generation
    scene.xml → Sionna RT → CIR, paths
              → Feature extraction (RT/PHY/MAC)
              → Zarr storage (chunked, compressed)
                    ↓
M3: Training
    Zarr → PyTorch Dataset → Transformer
         → Comet ML logging
                    ↓
M4: Physics Loss
    Precomputed radio maps → Differentiable lookup
                          → Physics consistency loss
```

### Zarr Dataset Schema

```
dataset.zarr/
├── rt_layer/
│   ├── path_gains      # [N, max_paths] complex64
│   ├── path_delays     # [N, max_paths] float32
│   └── rms_delay_spread # [N] float32
├── phy_fapi_layer/
│   ├── rsrp            # [N, num_cells] float32
│   ├── cqi             # [N, num_cells] int32
│   └── ri              # [N, num_cells] int32
├── mac_rrc_layer/
│   ├── serving_cell_id # [N] int32
│   └── timing_advance  # [N] int32
├── positions/
│   ├── ue_x, ue_y      # [N] float32 (ground truth)
├── radio_maps/         # [N, 7, H, W] float32
└── osm_maps/           # [N, 5, H, W] float32
```

---

## Why This Architecture?

| Design Choice | Rationale |
|---------------|-----------|
| **Dual maps** | Radio maps = physics priors; OSM maps = geometric constraints |
| **Precomputed maps** | O(1) lookup vs O(expensive) real-time ray tracing |
| **Coarse-to-fine** | Handles multi-modal posteriors, provides uncertainty |
| **Physics loss** | Regularizes predictions to be physically plausible |
| **Transformer** | Handles variable-length, irregularly-sampled sequences |

---

## Computational Complexity

| Component | Complexity | Typical Time |
|-----------|------------|--------------|
| Radio Encoder | $O(T^2 d)$ | ~5 ms |
| Map Encoder | $O(P^2 d)$ | ~10 ms |
| Cross-Attention | $O(T \cdot P \cdot d)$ | ~2 ms |
| Physics Loss | $O(B \cdot F)$ | ~5 ms |

Where $T$ = sequence length, $P$ = patches, $d$ = model dim, $B$ = batch, $F$ = features.

---

## References

- [Sionna RT Documentation](https://nvlabs.github.io/sionna/api/rt.html)
- [3GPP TS 38.215](https://www.3gpp.org/ftp/Specs/archive/38_series/38.215/) - Physical layer measurements
- [ITU-R P.2040](https://www.itu.int/rec/R-REC-P.2040) - Building material properties
