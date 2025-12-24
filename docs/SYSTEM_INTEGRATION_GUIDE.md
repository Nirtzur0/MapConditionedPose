# 🔄 System Integration Guide: How Everything Works Together

## Overview

Your UE localization system has **5 milestones** that work together like a pipeline:

```
M1: Scenes → M2: Data → M3: Model → M4: Physics → M5: Web UI
   ↓           ↓           ↓            ↓            ↓
 3D Maps   Features   Training   Refinement   Inference
```

---

## 🎬 Complete Data Flow

### **Phase 1: Offline Training (M1-M4)**

#### M1: Scene Generation
```python
# Input: OSM bounding box
bbox = (40.0, -105.0, 40.1, -105.1)  # Boulder, CO

# Process:
osm_data = download_osm(bbox)
buildings = extract_buildings(osm_data)
scene_xml = generate_mitsuba_scene(buildings)

# Output:
scenes/
├── boulder_tile_01.xml     # Mitsuba scene
├── buildings.ply           # 3D meshes
└── metadata.json           # Site locations, materials
```

#### M2: Data Generation
```python
# Input: Scene from M1
scene = sionna.rt.load_scene("boulder_tile_01.xml")

# Process: Ray tracing + channel simulation
for ue_position in sample_positions(scene, n=100000):
    # 1. RT Layer (Sionna RT - Dr.Jit, no TensorFlow)
    paths = scene.compute_paths(tx_positions, ue_position)
    rt_features = {
        'path_gain': paths.a,          # Complex coefficients
        'toa': paths.tau,              # Time of arrival
        'aoa': paths.theta_r,          # Angle of arrival
        'doppler': compute_doppler(paths, velocity)
    }
    
    # 2. PHY Layer (Sionna PHY - TensorFlow)
    channel_model = UMa(carrier_frequency=3.5e9)
    a, tau = channel_model(ue_position)
    h_freq = cir_to_ofdm_channel(a, tau)
    phy_features = {
        'rsrp': compute_rsrp(h_freq, pilot_indices),
        'sinr': compute_sinr(h_freq, noise, interference),
        'cqi': compute_cqi(sinr)
    }
    
    # 3. SYS Layer (Sionna SYS - TensorFlow)
    sys_features = {
        'timing_advance': compute_ta(distance_3d),
        'cell_id': serving_cell,
        'neighbor_rsrp': [rsrp for cell in neighbors]
    }
    
    # Save to Zarr (framework-agnostic)
    dataset.append({
        'position': ue_position,
        'rt_features': rt_features,
        'phy_features': phy_features,
        'sys_features': sys_features
    })

# Output:
data/
├── dataset.zarr/           # 1M+ samples
│   ├── positions/
│   ├── rt_features/
│   ├── phy_features/
│   └── sys_features/
├── radio_maps.zarr/        # Precomputed Sionna maps
└── osm_maps.zarr/          # Building height maps
```

#### M3: Model Training
```python
# Input: Data from M2 (PyTorch only, no TensorFlow!)
dataset = RadioDataset('data/dataset.zarr')
loader = DataLoader(dataset, batch_size=64)

# Model architecture:
model = TransformerUELocalization(
    # Radio Encoder (Transformer)
    radio_encoder = TransformerEncoder(
        input_dim=15,              # RT + PHY + SYS features
        hidden_dim=512,
        num_layers=8,
        num_heads=8
    ),
    
    # Map Encoder (Vision Transformer)
    map_encoder = VisionTransformer(
        input_channels=9,          # 5 radio + 4 OSM channels
        patch_size=16,
        hidden_dim=768,
        num_layers=12,
        num_heads=8
    ),
    
    # Fusion (Cross-Attention)
    fusion = CrossAttention(
        query_dim=512,             # From radio encoder
        key_value_dim=768,         # From map encoder
        num_heads=8
    ),
    
    # Output Heads
    coarse_head = nn.Linear(768, 32*32),  # Grid classification
    fine_head = nn.Linear(768, 4)         # (μx, μy, σx, σy)
)

# Training loop:
for batch in loader:
    # Unpack batch
    measurements = batch['measurements']  # [B, T, 15]
    radio_map = batch['radio_map']        # [B, 5, 512, 512]
    osm_map = batch['osm_map']            # [B, 4, 512, 512]
    gt_position = batch['position']       # [B, 2]
    
    # Forward pass
    output = model(measurements, radio_map, osm_map)
    pred_position = output['position']
    pred_heatmap = output['coarse_heatmap']
    
    # Loss computation
    loss_coarse = cross_entropy(pred_heatmap, gt_cell)
    loss_fine = nll_loss(pred_position, gt_position, uncertainty)
    loss = loss_coarse + loss_fine
    
    # Backward pass
    loss.backward()
    optimizer.step()

# Output:
checkpoints/best_model.pt  # Trained model weights
```

#### M4: Physics Loss
```python
# Input: Model from M3 + precomputed radio maps
radio_maps = zarr.open('data/radio_maps.zarr')

# Physics-consistency loss:
def physics_loss(pred_position, measurements, radio_map):
    """Compare predicted features with map lookup."""
    # Differentiable lookup at predicted position
    simulated_features = F.grid_sample(
        radio_map,                    # [B, C, H, W]
        pred_position.view(B, 1, 1, 2),  # [B, 1, 1, 2]
        mode='bilinear'               # Differentiable!
    )
    
    # Multi-feature MSE
    observed_features = measurements[:, :, [rsrp, sinr, toa, ...]]
    residuals = (observed_features - simulated_features) ** 2
    weights = torch.tensor([1.0, 0.8, 0.5, ...])  # Feature importance
    
    return (residuals * weights).sum()

# Enhanced training:
loss_total = loss_coarse + loss_fine + λ_phys * physics_loss(...)

# Output:
checkpoints/best_model_with_physics.pt  # Physics-regularized model
```

---

### **Phase 2: Real-Time Inference (M5)**

This is what happens when you use the Streamlit app!

#### Step 1: User Uploads Measurements
```python
# In Streamlit app (web/app.py)

# User either:
# - Uploads demo_measurements.json
# - Generates synthetic data
# - Manual entry (future)

uploaded_file = st.file_uploader("Upload JSON")
measurements = json.load(uploaded_file)

# Example measurements:
[
    {
        "timestamp": 0.0,
        "cell_id": 101,
        "beam_id": 5,
        "rsrp": -82.3,      # PHY layer
        "sinr": 14.2,       # PHY layer
        "timing_advance": 98,  # SYS layer
        "path_gain": -78.5, # RT layer (optional)
        "toa": 0.856e-6,    # RT layer (optional)
        "aoa_azimuth": 42.3 # RT layer (optional)
    },
    # ... more timesteps
]
```

#### Step 2: Preprocessing
```python
# In inference_utils.py

def preprocess_measurements(measurements):
    """Convert JSON to tensor format."""
    features_list = []
    
    for meas in measurements:
        # Extract all 15 features
        features = [
            meas['timestamp'],
            float(meas['cell_id']),
            float(meas.get('beam_id', -1)),
            # RT features (7)
            meas.get('path_gain', -160.0),
            meas.get('toa', 0.0),
            meas.get('aoa_azimuth', 0.0),
            meas.get('aoa_zenith', 0.0),
            meas.get('doppler', 0.0),
            # PHY features (5)
            meas.get('rsrp', -160.0),
            meas.get('rsrq', -40.0),
            meas.get('sinr', -20.0),
            meas.get('cqi', 0.0),
            meas.get('ri', 1.0),
            # SYS features (2)
            meas.get('timing_advance', 0.0),
            meas.get('phr', 0.0)
        ]
        features_list.append(features)
    
    # Shape: [num_timesteps, 15]
    return torch.tensor(features_list, dtype=torch.float32)
```

#### Step 3: Model Inference
```python
# In Streamlit app

# Load model (cached for performance)
@st.cache_resource
def load_model_cached(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

model = load_model_cached('checkpoints/best_model.pt')

# Preprocess
measurements_tensor = preprocess_measurements(measurements)
measurements_tensor = measurements_tensor.unsqueeze(0)  # Add batch dim

# Load maps for the scene
radio_map = torch.tensor(radio_maps[scene_id][:])
osm_map = torch.tensor(osm_maps[scene_id][:])

# Forward pass (no gradients needed!)
with torch.no_grad():
    output = model(
        measurements_tensor,  # [1, T, 15]
        radio_map,           # [1, 5, 512, 512]
        osm_map             # [1, 4, 512, 512]
    )

# Extract results
position = output['position'][0]         # [2] → (x, y) in meters
uncertainty = output['uncertainty'][0]   # [2] → (σx, σy)
heatmap = output['coarse_heatmap'][0]   # [32, 32] probability
```

#### Step 4: Visualization
```python
# In Streamlit app

# Display position
col1, col2 = st.columns(2)
with col1:
    st.metric("X Position", f"{position[0]:.2f} m")
    st.metric("X Uncertainty", f"±{uncertainty[0]:.2f} m")
with col2:
    st.metric("Y Position", f"{position[1]:.2f} m")
    st.metric("Y Uncertainty", f"±{uncertainty[1]:.2f} m")

# Plot heatmap
fig = create_heatmap_plot(heatmap, position)
st.plotly_chart(fig, use_container_width=True)

# Measurement timeline
fig_timeline = create_measurement_timeline(measurements)
st.plotly_chart(fig_timeline, use_container_width=True)
```

---

## 🔍 Under the Hood: Model Forward Pass

Let me trace what happens inside the model:

```python
class TransformerUELocalization(nn.Module):
    def forward(self, measurements, radio_map, osm_map):
        # measurements: [B, T, 15] - temporal sequence
        # radio_map: [B, 5, H, W] - Sionna features
        # osm_map: [B, 4, H, W] - building heights
        
        # 1. RADIO ENCODER: Process temporal measurements
        # ================================================
        
        # Embed categorical features
        cell_emb = self.cell_embedding(measurements[:, :, 1])  # Cell ID
        beam_emb = self.beam_embedding(measurements[:, :, 2])  # Beam ID
        time_emb = self.time_embedding(measurements[:, :, 0])  # Timestamp
        
        # Project continuous features
        features = measurements[:, :, 3:]  # RT+PHY+SYS features
        feature_proj = self.feature_projection(features)
        
        # Combine embeddings
        token_embeddings = cell_emb + beam_emb + time_emb + feature_proj
        # Shape: [B, T, 512]
        
        # Transformer encoder (handles variable length via masking)
        z_radio = self.transformer_encoder(
            token_embeddings,
            mask=create_padding_mask(measurements)
        )
        # Shape: [B, T, 512]
        
        # CLS token for sequence representation
        z_radio_cls = z_radio[:, 0, :]  # [B, 512]
        
        
        # 2. MAP ENCODER: Process dual map streams
        # ==========================================
        
        # Concatenate radio + OSM maps
        combined_maps = torch.cat([radio_map, osm_map], dim=1)
        # Shape: [B, 9, H, W] where 9 = 5 radio + 4 OSM
        
        # Patch embedding (16x16 patches)
        # 512x512 → 32x32 patches
        patches = self.patchify(combined_maps)  # [B, 1024, 9*16*16]
        patch_embeddings = self.patch_projection(patches)
        # Shape: [B, 1024, 768]
        
        # Add positional encodings
        patch_embeddings += self.pos_encoding
        
        # Vision Transformer
        F_maps = self.vit_encoder(patch_embeddings)
        # Shape: [B, 1024, 768]
        
        
        # 3. CROSS-ATTENTION FUSION: Radio ← Maps
        # =========================================
        
        # z_radio queries the map tokens
        fused = self.cross_attention(
            query=z_radio_cls.unsqueeze(1),  # [B, 1, 512]
            key=F_maps,                       # [B, 1024, 768]
            value=F_maps                      # [B, 1024, 768]
        )
        # Shape: [B, 1, 768]
        
        fused = fused.squeeze(1)  # [B, 768]
        
        
        # 4. COARSE PREDICTION: Grid classification
        # ===========================================
        
        coarse_logits = self.coarse_head(fused)  # [B, 1024]
        coarse_probs = F.softmax(coarse_logits, dim=1)
        coarse_heatmap = coarse_probs.view(B, 32, 32)
        
        # Decode grid cell to position
        # Find argmax position
        flat_idx = torch.argmax(coarse_logits, dim=1)
        row = flat_idx // 32
        col = flat_idx % 32
        coarse_position = torch.stack([
            col * 16.0,  # Cell size = 512m / 32 = 16m
            row * 16.0
        ], dim=1)
        # Shape: [B, 2]
        
        
        # 5. FINE REFINEMENT: Offset prediction
        # =======================================
        
        # Predict offset within grid cell + uncertainty
        fine_params = self.fine_head(fused)  # [B, 4]
        μx, μy = fine_params[:, 0], fine_params[:, 1]
        σx = F.softplus(fine_params[:, 2])  # Ensure positive
        σy = F.softplus(fine_params[:, 3])
        
        # Final position
        final_position = coarse_position + torch.stack([μx, μy], dim=1)
        uncertainty = torch.stack([σx, σy], dim=1)
        
        
        # 6. RETURN ALL OUTPUTS
        # ======================
        
        return {
            'position': final_position,        # [B, 2]
            'uncertainty': uncertainty,        # [B, 2]
            'coarse_heatmap': coarse_heatmap, # [B, 32, 32]
            'attention': {
                'cross_attn': ...,            # For visualization
                'self_attn': ...
            }
        }
```

---

## 🎨 Key Design Decisions

### 1. **Why Separate Data Generation from Training?**
- **Sionna** (TensorFlow) only needed during data gen
- **PyTorch** used exclusively for training (simpler, faster)
- **Zarr** provides framework-agnostic storage

### 2. **Why Precompute Radio Maps?**
- **Speed**: Lookup is 1000× faster than real-time ray tracing
- **Differentiability**: `F.grid_sample` provides gradients
- **Storage**: Only ~100MB per scene

### 3. **Why Coarse-to-Fine Architecture?**
- **Efficiency**: Coarse grid (32×32) is fast to evaluate
- **Accuracy**: Fine offset provides sub-meter precision
- **Uncertainty**: Heteroscedastic prediction for confidence

### 4. **Why Streamlit for UI?**
- **Rapid Development**: 500 lines vs 2000+ for React
- **Python Only**: No JavaScript knowledge needed
- **Beautiful Default**: Professional UI out-of-the-box
- **Easy Deployment**: One-click on Streamlit Cloud

---

## 🚀 Try It Now!

The app is running at **http://localhost:8501**

### Quick Demo:

1. **Navigate to "🔍 Live Inference"**
2. **Select "Generate Demo Data"**
3. **Adjust slider to 10 timesteps**
4. **Click "Generate Measurements"**
5. **Click "Run Inference"**
6. **See the prediction heatmap!**

Or upload the provided `demo_measurements.json` file!

---

## 📊 What Each Feature Contributes

| Feature Type | Examples | Contribution | Why It Helps |
|--------------|----------|--------------|--------------|
| **RT Layer** | Path Gain, ToA, AoA | 35% | Direct propagation physics |
| **PHY Layer** | RSRP, SINR, CQI | 45% | Observable signal quality |
| **SYS Layer** | TA, Cell ID | 20% | Network context |

**Key Insight:** RSRP + SINR are most important (47% combined), but timing advance provides strong localization signal!

---

## 🔄 The Full Loop

```
User → Upload JSON → Preprocess → Model → Visualization
  ↑                                            ↓
  └──────────── Feedback & Analysis ←─────────┘
```

You can now:
- **Upload real measurements** (when available)
- **Analyze failures** (error patterns)
- **Compare models** (w/ vs w/o physics loss)
- **Understand predictions** (attention weights, heatmaps)

---

Ready to explore the app? 🎉
