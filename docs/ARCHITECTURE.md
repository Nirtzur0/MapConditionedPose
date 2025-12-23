# Transformer UE Localization: Technical Architecture

**Domain:** Statistical Signal Processing, Bayesian Inference, and Deep Learning for Wireless Communications  
**Framework:** Sionna RT (Differentiable Ray Tracing) + PyTorch + Dr.Jit  
**Last Updated:** December 23, 2025

---

## Mission Statement

Design and implement an end-to-end Neural Network (NN) architecture for User Equipment (UE) localization that transitions from a purely data-driven approach to a **physics-conditioned inference** engine. By leveraging high-fidelity electromagnetic (EM) simulations via Sionna RT as a "Digital Twin" generator, we aim to solve the inverse problem:

$$p(x | M_t, R_{\text{maps}}, G_{\text{maps}})$$

where:
- $M_t$: Sparse temporal measurement sequence
- $R_{\text{maps}}$: Precomputed Sionna radio maps (physics priors)
- $G_{\text{maps}}$: OSM building/terrain maps (geometric constraints)
- $x$: UE position

---

## 1. Problem Definition & Formalization

### 1.1 Observable Manifold ($\mathcal{M}_t$)

The input at time $t$ is a hierarchical set of multi-layer tokens $M_t = \{m_1(t_1), m_2(t_2), \ldots, m_N(t_T)\}$, where each token represents measurements from **actual protocol stack layers** as they would appear in a real 5G/LTE network.

**Architecture:** We map the full protocol stack to positioning features:
- **Layer 1 (RT):** Simulation ground truth - propagation physics
- **Layer 2 (PHY/FAPI):** L1 measurements - what UE PHY reports
- **Layer 3 (MAC/RRC):** L2/L3 measurements - protocol layer observations  
- **Layer 4 (RLC):** Optional link statistics

#### Layer 1: RT (Ray Tracer) - Propagation Physics Ground Truth

* **Path Gain:** $\text{PG}_i$, quantized (1 dB steps) with Shadow Fading $\sigma_{\text{SF}}$
* **Time of Arrival:** $\tau_i$, where $\tau_{\text{NLOS}} = \tau_{\text{LOS}} + \Delta\tau$ with NLoS bias
* **Angle of Arrival/Departure:** $(\theta_i, \phi_i)$, $(\theta_i^{\text{Tx}}, \phi_i^{\text{Tx}})$ derived from array response
* **Doppler Shifts:** $f_{D,i}$ from path velocities
* **Multipath Sparsity:** RMS Delay Spread $\tau_{\text{RMS}}$ from Power Delay Profile (PDP)

#### Layer 2: PHY/FAPI (L1) - Channel & Physical Layer Features

**FAPI Measurements (from UE PHY → Network):**
* **RSRP (Reference Signal Received Power):** Per cell, per beam (L1-RSRP for beam-specific)
* **RSRQ (Reference Signal Received Quality):** Per cell, quality indicator
* **RSSI (Received Signal Strength Indicator):** Wideband power including OFDM effects
* **SINR (Signal-to-Interference-plus-Noise Ratio):** Per resource block, per antenna
* **SNR (Signal-to-Noise Ratio):** Per antenna element, thermal noise limited

**Channel State Information (CSI Reports via FAPI):**
* **CQI (Channel Quality Indicator):** 0-15, maps to modulation/coding scheme
* **RI (Rank Indicator):** MIMO rank (1-8), number of spatial streams
* **PMI (Precoding Matrix Indicator):** Codebook index for beamforming
* **CSI Matrix:** $\mathbf{H}$ compressed (rank, condition number, singular values)
* **L1-RSRP per beam:** Beam-level measurements for beam management

**Channel Characteristics:**
* **Channel Impulse Response (CIR):** Tap magnitudes, delays, compressed summaries
* **Power Delay Profile (PDP):** Multipath power distribution
* **Doppler Spectrum:** Frequency spread from mobility

#### Layer 3: MAC/RRC (L2/L3) - Protocol & Network Features

**MAC Layer Measurements:**
* **Timing Advance (TA):** Quantized (16 Ts units in 5G NR), distance proxy
* **Timing Advance Command (TAC):** Adjustment values, temporal tracking
* **Uplink Timing Error:** Fine-grained timing offset
* **Power Headroom Report (PHR):** UE transmit power margin
* **HARQ ACK/NACK Statistics:** Block error patterns, link quality
* **Buffer Status Report (BSR):** Traffic load (indirect mobility indicator)
* **Scheduling Request (SR) Frequency:** Access pattern statistics

**RRC Layer Measurements (Measurement Reports):**
* **Serving Cell ID (PCI):** Physical Cell Identity
* **Neighbor Cell List:** Top-K cells (typically K=8), with PCI
* **RSRP per neighbor:** Measurement report values, 1 dB quantization
* **RSRQ per neighbor:** Quality metric per neighbor cell
* **RSRP/RSRQ Ratios:** Serving vs neighbor signal strength
* **Handover Decision Metrics:** Cell reselection criteria, thresholds
* **Time-to-Trigger (TTT):** Handover hysteresis timing
* **Event-based Triggers:** A3 (neighbor better), A5 (serving threshold)

**RRC Connection Context:**
* **UE Capability Info:** RF bands, antenna configuration, positioning support
* **SRS (Sounding Reference Signal) Configuration:** Uplink positioning signals
* **PRS (Positioning Reference Signal) Configuration:** DL-TDOA positioning

**System-Level Performance (Physical Layer Abstraction):**
* **Throughput Estimate:** $R_{\text{est}}$ from effective SINR mapping
* **Block Error Rate (BLER):** Predicted from SINR and MCS
* **Spectral Efficiency:** Bits/s/Hz achieved
* **Frame Error Rate (FER):** Transport block success rate

#### Layer 4: RLC (Radio Link Control) - Link Statistics (Optional)

**RLC Layer Metrics (if available from network stack):**
* **RTT (Round-Trip Time) Estimates:** From RLC ACK timing
* **Packet Delivery Statistics:** Success rate, latency distribution
* **Retransmission Counts:** RLC ARQ statistics, link quality indicator
* **Throughput per Bearer:** Application-level data rates
* **Jitter/Latency Variance:** Statistical moments of delay

**Note:** RLC parameters are typically less directly positioning-relevant but can provide:
- Mobility indicators (high retransmissions → challenging channel)
- Connection stability metrics (complement to MAC/PHY measurements)
- Temporal correlation features (persistent bad link → static NLoS)

---

### Protocol Stack to Observable Manifold Mapping

**Complete Feature Map (What You Get from Real Network):**

| Protocol Layer | Parameter | Positioning Relevance | Typical Availability | Priority |
|----------------|-----------|----------------------|---------------------|----------|
| **FAPI/PHY (L1)** | | | | |
| | RSRP (serving) | ★★★★★ Direct power | Always | Critical |
| | RSRP (neighbors) | ★★★★★ Multilateration | Always | Critical |
| | RSRQ | ★★★☆☆ Quality, interference | Always | Medium |
| | RSSI | ★★★★☆ Wideband power | Always | High |
| | L1-RSRP per beam | ★★★★★ Beam-level AoA proxy | 5G NR FR2 | High |
| | SINR | ★★★★☆ Link quality, interference | Often | High |
| | CQI | ★★★☆☆ Channel quality proxy | Always | Medium |
| | RI (Rank) | ★★☆☆☆ MIMO spatial structure | Often | Low |
| | PMI | ★★☆☆☆ Precoding, spatial info | Often | Low |
| | CSI matrix | ★★★★☆ Full channel state | Rarely (compressed) | High |
| **MAC (L2)** | | | | |
| | Timing Advance (TA) | ★★★★★ Distance proxy | Always | Critical |
| | TA Command (TAC) | ★★★★☆ Temporal TA tracking | Often | High |
| | PHR (Power Headroom) | ★★☆☆☆ Indirect path loss | Often | Low |
| | HARQ ACK/NACK stats | ★★☆☆☆ Link quality | Sometimes | Low |
| | BSR | ★☆☆☆☆ Traffic pattern | Rarely | Very Low |
| **RRC (L3)** | | | | |
| | Serving Cell ID (PCI) | ★★★★★ Cell identity | Always | Critical |
| | Neighbor Cell IDs | ★★★★★ Multi-cell context | Always | Critical |
| | Measurement Reports | ★★★★★ Comprehensive measurements | Always | Critical |
| | Handover triggers | ★★★☆☆ Mobility pattern | Often | Medium |
| | SRS config | ★★★★☆ Uplink positioning setup | 5G positioning | High |
| | PRS config | ★★★★☆ DL-TDOA positioning setup | 5G positioning | High |
| **RLC (L2)** | | | | |
| | RTT estimates | ★★★☆☆ Timing proxy | Rarely | Medium |
| | Retransmission counts | ★★☆☆☆ Link stability | Sometimes | Low |
| | Per-bearer throughput | ★☆☆☆☆ Indirect quality | Rarely | Very Low |
| **Ray Tracing (Simulation)** | | | | |
| | Path Gain | ★★★★★ Ground truth power | Simulation | Critical |
| | ToA per path | ★★★★★ Multipath timing | Simulation | Critical |
| | AoA/AoD per path | ★★★★★ Spatial signatures | Simulation | Critical |
| | Doppler per path | ★★★☆☆ Velocity, multipath | Simulation | Medium |
| | RMS Delay Spread | ★★★★☆ Multipath richness | Simulation | High |

**Priority Legend:**
- **Critical:** Essential for positioning, always use if available
- **High:** Significantly improves accuracy, use when available
- **Medium:** Provides complementary information, include if feasible
- **Low:** Marginal benefit, optional
- **Very Low:** Minimal positioning value, skip unless doing extensive ablations

### 1.3 Sionna Capability Mapping

Based on analysis of Sionna's codebase (TR38901, RT, PHY, SYS modules), here's what can be **directly obtained** vs **requires custom implementation**:

**✅ Direct from Sionna TR38901/RT:**
- **Rays:** `rays.delays`, `rays.powers`, `rays.aoa`, `rays.aod`, `rays.zoa`, `rays.zod`, `rays.xpr`
- **LSP (Large Scale Parameters):** `lsp.ds` (RMS delay spread), `lsp.asd/asa/zsa/zsd` (angle spreads), `lsp.sf` (shadow fading), `lsp.k_factor` (Rician K)
- **Channel Coefficients:** `h` (complex path gains, shape `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`), `tau` (path delays)
- **Topology:** `velocities`, `los` (bool), `distance_2d`, `distance_3d`, `indoor` state, `orientations`
- **Basic Pathloss:** `scenario.basic_pathloss`, `los_probability`
- **Doppler:** Computed from velocities + angles via utilities

**✅ Direct from Sionna PHY/SYS:**
- **Channel Utilities:** `cir_to_ofdm_channel`, `cir_to_time_channel`, `time_to_ofdm_channel`
- **Antenna Arrays:** `PanelArray`, `AntennaElement` configurations
- **System-Level:** `get_pathloss` (from frequency response), `is_scheduled_in_slot`
- **Channel Models:** UMi/UMa/RMa system-level scenarios, CDL/TDL link-level models

**🔧 Custom Implementation Required (compute from Sionna outputs):**

**PHY/FAPI Layer (from channel `h`, `tau`):**
- **RSRP:** Compute from $|\mathbf{h}|^2$ per resource element, average across pilot symbols
- **RSRQ:** $\text{RSRQ} = \frac{N \times \text{RSRP}}{\text{RSSI}}$ (requires OFDM grid simulation for interference)
- **RSSI:** Wideband received power (sum across subcarriers)
- **L1-RSRP per beam:** Requires beam-specific channel realization + SSB/CSI-RS processing
- **CQI (0-15):** Map SINR → CQI via 3GPP tables (link abstraction)
- **RI (Rank Indicator):** SVD of channel matrix `h`, threshold eigenvalues
- **PMI (Precoding Matrix Index):** Codebook selection via capacity/SINR maximization
- **CSI feedback:** Quantize eigenvectors/eigenvalues per 3GPP Type I/II CSI

**MAC/RRC Layer (from topology + system-level):**
- **TA (Timing Advance):** $\text{TA} = \frac{2 \times \text{distance\_3d}}{c}$, quantized to 16 Ts (0.5 μs steps)
- **TAC (TA Command):** Track $\Delta \text{TA}$ over time
- **PHR (Power Headroom):** $\text{PHR} = P_{\max} - P_{\text{TX}}$ (requires power control logic)
- **HARQ ACK/NACK:** Link abstraction via BLER(SINR) lookup tables + random sampling
- **BSR (Buffer Status):** Synthetic traffic model (Poisson arrivals, queue simulation)
- **Measurement Reports (A3/A4/A5):** Aggregate RSRP/RSRQ → threshold logic per 3GPP 36.331
- **Cell Selection Criteria:** Rank cells via Srxlev/Squal formulas (3GPP 38.304)

**RLC Layer (from SYS utilities):**
- **PDU Counts:** Track successful/retransmitted packets via ARQ simulation
- **Throughput:** Shannon capacity $C = B \log_2(1 + \text{SINR})$ with system-level overhead
- **Latency:** Queuing delay (M/M/1) + transmission time + HARQ RTT

**Measurement Realism:**
- **Quantization:** Round RSRP/RSRQ to 1 dB, TA to 16 Ts, CQI to integers (0-15)
- **Neighbor List Truncation:** Keep top-K=8 cells by RSRP (3GPP measurement reporting limits)
- **Feature Dropout:** Randomly drop measurements per protocol timing (80-480ms for RRC, 5-40ms for L1)
- **Temporal Irregularity:** Sample measurement periods from realistic distributions

**Implementation Strategy:**
All custom features are **precomputed during data generation** (M2) using PyTorch/NumPy on Sionna's outputs, then stored in Zarr. The training loop (M3) loads these as static tensors—no Sionna/TensorFlow at training time.

---

**Key Property:** Each token $m_i(t_j)$ combines features from RT (Layer 1), PHY/FAPI (Layer 2), MAC/RRC (Layer 3), and optionally RLC (Layer 4) into a unified observable manifold for the transformer.

**Measurement Realism Constraints:**
- Quantization: 1 dB steps for power measurements, TA quantization (16 Ts in 5G NR)
- Spatial dropout: Neighbor list truncation (top-K=8 cells only per 3GPP specs)
- Feature dropout: Missing tokens (CQI not reported, CSI compression, beam measurements only in FR2)
- Temporal irregularity: Variable measurement periods (80-480ms for RRC measurements, faster for L1)
- Protocol-specific availability: L1-RSRP only in 5G NR FR2, SRS/PRS only with positioning support

**Advanced 5G NR Positioning Features (if UE/network supports):**
- **DL-TDOA (Downlink Time Difference of Arrival):** PRS-based, multiple gNB timing
- **UL-TDOA (Uplink TDOA):** SRS-based, network-side measurements
- **Multi-RTT (Round-Trip Time):** Combined DL/UL timing for ranging
- **DL-AoD (Downlink Angle of Departure):** From gNB antenna array
- **UL-AoA (Uplink Angle of Arrival):** Measured at gNB from SRS
- **Carrier Phase Measurements:** Sub-wavelength accuracy (advanced feature)

### 1.2 Objective Function

The model approximates the posterior $p(x | M_t, R_{\text{maps}}, G_{\text{maps}})$ as a hierarchical density:

1. **Coarse:** Categorical distribution over discretized grid $p(\text{cell} | \cdot)$
2. **Fine:** Gaussian Mixture Model (GMM) or Heteroscedastic Regressor for sub-grid refinement: $p(x | \text{cell}, \cdot) = \mathcal{N}(\mu, \Sigma)$

---

## 2. Network Input Architecture

The network receives **three complementary input modalities** for both training and inference:

### A) Sparse Temporal Measurement Sequence

**Characteristics:**
- **Temporal:** Variable-length sequences with irregular timestamps
- **Spatial Sparsity:** Not all cells/beams report at each time (neighbor list truncation)
- **Feature Sparsity:** Individual features may be missing (no AoA, missing TA, etc.)

**Structure:**
- Each token $m_i(t_j)$ contains RT+PHY+SYS layer features
- Requires masking mechanism for missing tokens/features
- Permutation invariant to cell/beam order (handled via Set Transformer)

### B) Precomputed Sionna Radio Maps (Physics Priors)

**Content:**
- **RT Layer Grids:** Path Gain, ToA, AoA per BS
- **PHY Layer Grids:** SNR, SINR, CSI metrics
- **SYS Layer Grids:** Throughput, BLER estimates

**Properties:**
- Dense spatial representation (no sparsity)
- Resolution: 1-2m per pixel
- Coverage: Full scene (e.g., 512m × 512m)
- **Used in:** Both training and inference (no distribution shift)

**Generation:** Use Sionna's `RadioMapSolver` to precompute comprehensive multi-feature maps across all frequency bands and BS configurations.

### C) OSM Building Maps (Geometric Context)

**Content:**
- Building heights (DSM/DTM)
- Material properties (permittivity/conductivity classes)
- Building footprint masks
- Road network masks
- Terrain elevation

**Properties:**
- Dense spatial representation
- Resolution: Matched to radio maps (1-2m per pixel)
- Provides geometric feasibility constraints (e.g., "UE must be on streets, not inside buildings")
- **Used in:** Both training and inference

### Complementary Map Benefits

**Sionna Radio Maps (Physics):**
- ✅ Encode signal propagation physics
- ✅ Multi-path effects, shadowing, reflections
- ✅ Frequency-specific behavior
- ✅ Multi-layer features (RT+PHY+SYS)

**OSM Building Maps (Geometry):**
- ✅ Encode geometric feasibility
- ✅ "UE must be on streets, not inside buildings"
- ✅ Material properties affect propagation
- ✅ Topological constraints (road networks)

**Together:** Radio maps provide "signal looks like this here" while OSM maps enforce "UE can physically be here". This fusion resolves ambiguities neither can solve alone (e.g., in NLOS, radio signal is diffuse but OSM constrains UE to street/sidewalk).

---

## 3. Neural Network Architecture

### 3.1 Radio Encoder (Temporal Set Transformer)

**Purpose:** Process sparse temporal measurement sequences

**Token Design:**
Each observed (cell, beam, time) tuple becomes a token:
$$\text{token}_i = \text{Embed}(\text{cell\_id}) + \text{Embed}(\text{beam\_id}) + \text{Embed}(t_i) + \text{Features}(m_i)$$

where Features$(m_i)$ includes RT+PHY+SYS layer measurements.

**Architecture:**
- 6-12 transformer encoder layers
- Self-attention with masking for missing tokens
- Handles variable-length sequences
- **Output:** Radio embedding $z_{\text{radio}}$ (CLS token) + optional per-token outputs

**Key Property:** Permutation invariant to cell/beam order.

### 3.2 Dual Map Encoder

**Purpose:** Process two complementary dense map representations

**Two Input Streams:**

1. **Sionna Radio Maps Stream**
   - Multi-channel input: [PG, ToA, AoA, SNR, SINR, Throughput, BLER]
   - Encodes physics-based signal features

2. **OSM Building Maps Stream**
   - Multi-channel input: [height, materials, footprints, roads, terrain]
   - Encodes geometric/topological features

**Architecture Options:**
- **Option A:** Separate encoders → concatenate
- **Option B:** Early fusion → single encoder
- **Option C:** Cross-attention between streams

**Backbone Choices:**
- Vision Transformer (ViT): Patch embedding + transformer encoder
- CNN+Transformer: ResNet/ConvNeXt backbone → flatten → transformer
- Hybrid: Hierarchical multi-resolution encoding

**Outputs:**
- $F_{\text{radio\_map}}$: Spatial features from physics (grid of tokens)
- $F_{\text{osm\_map}}$: Spatial features from geometry (grid of tokens)

### 3.3 Spatio-Temporal Cross-Attention Fusion

**Mechanism:**
```
Query: z_radio (from temporal measurements)
Keys/Values: [F_radio_map ; F_osm_map] (from dual maps)

Cross-Attention → Fused representation → Position heatmap
```

**Multi-Head Cross-Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q = W_Q z_{\text{radio}}$ (radio query)
- $K, V = W_K [F_{\text{radio\_map}}; F_{\text{osm\_map}}]$ (map keys/values)

**Outputs:**
- **Coarse:** Probability over grid cells $p(\text{cell} | \cdot)$
- **Fine:** Sub-grid offset $(\Delta x, \Delta y)$ or GMM parameters

### 3.4 Output Heads

**Coarse Head (Classification):**
- Heatmap over discretized grid cells
- Cross-entropy loss with true cell index
- Provides spatial uncertainty visualization

**Fine Head (Regression):**
- Conditioned on top-K coarse cells
- Predicts continuous offset within cell
- Options:
  - Simple regression: $(\Delta x, \Delta y)$ with Huber loss
  - Heteroscedastic: $(\mu_x, \mu_y, \sigma_x, \sigma_y)$ with NLL loss
  - Mixture: GMM with $M$ components for multi-modal distributions

---

## 4. Differentiable Physics Integration

### 4.1 Physics-Consistency Loss with Sionna Maps

We define a **Physics-Consistency Loss** using precomputed Sionna radio maps to regularize the latent space:

$$\mathcal{L}_{\text{phys}}(\hat{x}) = \sum_{i=1}^{N} w_i \left\| m_i^{\text{obs}} - \text{Lookup}_{R_{\text{maps}}}(\hat{x}, \text{feature}_i) \right\|^2$$

where:
- $\hat{x}$: Predicted UE location
- $m_i^{\text{obs}}$: Observed measurement for feature $i$
- $\text{Lookup}_{R_{\text{maps}}}(\cdot)$: Differentiable bilinear interpolation in precomputed map
- $w_i$: Feature-specific weights (based on measurement reliability)
- Features include: Path Gain, ToA, AoA, SNR, SINR, Throughput, BLER

### 4.2 Training Mechanism

1. **Precompute Dual Maps:**
   - **Sionna Radio Maps:** Generate comprehensive multi-layer radio maps (RT+PHY+SYS features) for each scene using `RadioMapSolver`
   - **OSM Building Maps:** Extract and process geographic/geometric data

2. **Network Forward Pass:**
   - Input: Sparse temporal measurements $M_t$ + Precomputed Radio Maps $R_{\text{maps}}$ + OSM Maps $G_{\text{maps}}$
   - Output: Predicted location $\hat{x}$

3. **Differentiable Lookup:** For predicted location $\hat{x}$, perform bilinear interpolation in precomputed Sionna radio maps

4. **Multi-Feature Physics Loss:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{localization}} + \lambda \mathcal{L}_{\text{phys}}$$

where $\mathcal{L}_{\text{localization}}$ is the standard supervised loss (coarse + fine) and $\lambda$ controls physics regularization strength.

### 4.3 Inference Mechanism

**Input:** Same as training - sparse measurements + dual maps (radio + OSM)

**Process:**
1. Forward pass with all three inputs (identical architecture to training)
2. Predict position $\hat{x}$
3. **Optional refinement:** Query precomputed Sionna maps for position validation
4. **Multi-feature matching:** Compare observed $M_t$ vs map-predicted features
5. **Confidence estimation:** Residual magnitude indicates prediction reliability

**Key Advantage:** Training and inference use identical inputs and architecture - no distribution shift.

### 4.4 Implementation with Dr.Jit

Since Sionna `RadioMapSolver` outputs are available via Dr.Jit's `@dr.wrap` interop, we can backpropagate the residual error between observed and simulated multi-layer features at predicted location $\hat{x}$ through to the positioning network.

**Gradient Flow:**
```
PyTorch Network → Predicted x̂ → Dr.Jit Bilinear Lookup → 
Sionna RadioMap Values → Loss → Gradients back through x̂
```

This enables end-to-end optimization of the positioning network using physics-validated predictions.

---

## 5. Synthetic Data Generation

### 5.1 Scene Representation

Scenes are modeled as triplet (Vertices, Faces, Site configurations) using Mitsuba 3 format compatible with Sionna RT.

**Domain Randomization (DR):** To mitigate the Reality Gap, we randomize:

* **Material Properties:**
  - Complex permittivity $\epsilon_r$ per building/material class
  - Conductivity $\sigma$ with plausible bounds
  - Spatial variation within material classes

* **Measurement Noise Models:**
  - Quantization (1 dB steps for power, TA quantization)
  - Shadow fading (spatial and temporal correlation)
  - NLoS bias in timing measurements
  - Measurement dropout (neighbor list truncation)

* **OFDM Parameters:**
  - Subcarrier spacing variation
  - Cyclic prefix lengths
  - Interference patterns

* **Network Protocol Parameters:**
  - Handover thresholds
  - TA quantization schemes
  - Cell selection criteria

### 5.2 Sionna Multi-Layer Data Generation

**RadioMapSolver:** Generate comprehensive precomputed maps with RT, PHY, and SYS layer features.

**Multi-Feature Grids:**
- **RT:** Path Gain, ToA, AoA, AoD, Doppler, RMS Delay Spread
- **PHY:** CIR summaries, SNR, SINR, CSI metrics, RSSI
- **SYS:** Throughput, BLER, Handover metrics, Cell IDs, RSRP/RSRQ ratios

**UE Sampling:** Non-uniform sampling proportional to coverage probability using Sionna's `sample_positions()` with SINR/RSS constraints and distance limits. This ensures training data concentrates where UEs exist in practice.

### 5.3 Realistic Measurement Report Generation

From RT outputs (paths, CIR, per-site gain), synthesize "network observable" features:

**Per UE Report:**
- Serving cell/sector ID (+ beam ID in 5G)
- Neighbor list metrics (top-K cells, top-B beams): RSRP/RSRQ/SINR
- Timing features: TA / RTT / ToA / TDoA (with NLoS bias)
- Optional: AoA (or beam/sector), Doppler, delay spread, K-factor proxy

**Temporal Sequences:**
- Short window (5-20 reports) with timestamps
- Irregular sampling (realistic measurement rates)
- Missing features (dropout, unavailability)

---

## 6. Technical Stack

* **Simulation:** `sionna-rt` (v0.14+) utilizing the Mitsuba 3 backend for differentiable rendering
* **Differentiability:** `drjit` for gradient propagation through Eikonal solvers or Path Tracing kernels
* **Geometry:** `osmnx` for Graph-based GIS extraction; `shapely` for intersection and footprint processing
* **ML Framework:** `torch` + `pytorch-lightning` for distributed training
* **Data Storage:** `zarr` (chunked, cloud-friendly) or `parquet` for tabular features
* **Visualization:** `wandb` or `tensorboard` for experiment tracking

---

## 7. Key Design Decisions & Rationale

### 7.1 Why Three Input Modalities?

**Sparse Measurements Alone:**
- ❌ Highly ambiguous in NLOS scenarios
- ❌ Limited spatial resolution
- ❌ Prone to multipath confusion

**Radio Maps Alone:**
- ❌ Don't account for temporal variations
- ❌ Precomputed, may not match observed conditions exactly
- ❌ No geometric feasibility enforcement

**OSM Maps Alone:**
- ❌ Don't encode signal propagation
- ❌ Can't resolve ambiguities based on RF measurements

**All Three Together:**
- ✅ Measurements provide temporal context and real conditions
- ✅ Radio maps provide physics priors and propagation constraints
- ✅ OSM maps provide geometric feasibility and topological structure
- ✅ Fusion resolves ambiguities none can solve individually

### 7.2 Why Precomputed Maps vs Real-Time RT?

**Precomputed Approach:**
- ✅ Fast: bilinear lookup is O(1) vs full ray tracing
- ✅ Scalable: compute once, use for all training/inference
- ✅ Differentiable: smooth gradients through interpolation
- ✅ Consistent: same physics model in training and inference

**Real-Time RT:**
- ❌ Expensive: minutes per sample with full path tracing
- ❌ Non-scalable: prohibitive for large datasets
- ✅ Flexible: can adapt to exact conditions

**Hybrid Solution:** Use precomputed maps for main training/inference, optional real-time RT refinement for high-stakes predictions.

### 7.3 Why Coarse-to-Fine Output?

**Single-Stage Regression:**
- ❌ Overconfident when wrong
- ❌ No uncertainty quantification
- ❌ Struggles with multi-modal posteriors

**Coarse-to-Fine:**
- ✅ Heatmap provides spatial uncertainty visualization
- ✅ Fine refinement focuses computation on likely regions
- ✅ Naturally handles multi-modal distributions (multiple peaks)
- ✅ Interpretable: can inspect coarse predictions

---

## 8. Critical Challenges & Mitigation

### 8.1 Spatial Overfitting

**Problem:** Models memorize specific building layouts of training tiles.

**Mitigation:**
- Mandatory validation on "Unseen Cities"
- Domain randomization of materials and geometry
- Training on diverse city tiles (5-10+ different urban morphologies)
- Architecture designed for generalization (attention, not position-specific)

### 8.2 NLoS Ambiguity

**Problem:** Purely diffuse radio signals are spatially ambiguous.

**Mitigation:**
- Dual map input: Radio maps provide propagation, OSM maps provide geometric constraints
- Multi-feature validation: cross-check Path Gain, ToA, AoA consistency
- Uncertainty quantification: flag high-ambiguity predictions

### 8.3 Map Alignment

**Problem:** Radio maps and OSM maps must be precisely co-registered.

**Mitigation:**
- Automated coordinate system alignment in data pipeline
- Consistent spatial reference system (e.g., UTM projection)
- Validation: verify site locations match between maps

### 8.4 Temporal Sparsity

**Problem:** Highly variable measurement rates and missing features.

**Mitigation:**
- Self-attention with masking for missing tokens
- Token-level feature masking for incomplete measurements
- Training with variable sequence lengths and dropout

### 8.5 Computational Cost

**Problem:** Processing dual dense maps + sparse temporal sequences.

**Mitigation:**
- Hierarchical/multi-resolution map encoding
- Coarse-to-fine: process full map at low-res, high-res only at candidates
- Efficient ViT architectures with patch-based processing
- Precomputed maps eliminate runtime RT overhead

---

## 9. Skepticism & Validation Requirements

### 9.1 Sim-Only Generalization

**Concern:** Training purely on synthetic data may not transfer to real deployments.

**Evidence Needed:**
- Domain randomization ablations showing robustness to material/noise variations
- Validation on diverse city types (dense urban, suburban, rural)
- Comparison to classical methods (triangulation, fingerprinting) as sanity check

### 9.2 Physics Loss Value

**Concern:** Physics loss may be expensive without clear benefit.

**Evidence Needed:**
- Ablation: no physics loss vs precomputed map loss vs full RT loss
- Quantify: accuracy gain vs computational cost
- Analysis: when does physics loss help most (high ambiguity scenarios?)

### 9.3 Map Dependency

**Concern:** Model may rely too heavily on maps, fail when maps are outdated.

**Evidence Needed:**
- Robustness testing with deliberately outdated/incorrect maps
- Ablation: performance degradation vs map error magnitude
- Failure mode analysis: how does model behave with bad maps?

### 9.4 Transformer Justification

**Concern:** Transformer may be overkill for simple measurement vectors.

**Evidence Needed:**
- Baseline comparisons: MLP, CNN, simpler attention
- Ablation: benefit of self-attention vs feed-forward
- Scalability: performance vs number of neighbor cells, time steps

---

## 10. Future Extensions

### 10.1 Real-World Deployment

- Fine-tuning on small real measurement datasets
- Online learning / adaptation to site-specific characteristics
- Calibration of measurement noise models from real data

### 10.2 Advanced Physics Integration

- Gradient-based refinement at inference (hybrid approach)
- Meta-learning for fast adaptation to new scenes
- Learning material properties from measurements

### 10.3 Multi-Modal Sensing

- Integration with GNSS (when available)
- IMU-based motion priors
- Visual/LiDAR for enhanced geometric context

### 10.4 System-Level Features

- Joint localization and tracking (temporal filtering)
- Multi-user collaborative positioning
- Network optimization (joint positioning and resource allocation)

---

## Summary

This architecture provides a principled approach to UE localization by:

1. **Leveraging physics:** Sionna RT provides validated propagation models
2. **Dual map conditioning:** Combining radio and geometric priors resolves ambiguities
3. **Sparse temporal modeling:** Transformers handle irregular, incomplete measurements
4. **End-to-end differentiability:** Physics loss regularizes learning
5. **Training-inference consistency:** Same inputs and architecture eliminate distribution shift

The system is designed for generalization through domain randomization, validated on unseen cities, and scalable through precomputed maps and efficient architectures.
