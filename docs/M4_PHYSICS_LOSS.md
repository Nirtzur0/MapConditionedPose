# M4: Differentiable Physics Regularization

**Status**: Complete (Implementation + Tests)
**Tests**: 17/17 passing

## Overview

M4 implements physics-consistency loss using precomputed Sionna radio maps with differentiable bilinear interpolation. This allows the model to learn from both supervised labels (ground truth positions) and physics-based constraints (electromagnetic propagation).

## Architecture

### Components

1. **Differentiable Lookup** (`src/physics_loss/differentiable_lookup.py`)
 - Bilinear interpolation using `F.grid_sample`
 - Fully differentiable w.r.t. predicted positions
 - Supports batch and multi-point sampling

2. **Physics Loss** (`src/physics_loss/physics_loss.py`)
 - Multi-feature weighted MSE loss
 - Configurable feature weights (path_gain: 1.0, ToA: 0.5, AoA: 0.3, SNR/SINR: 0.8, throughput/BLER: 0.2)
 - Supports MSE or Huber loss
 - Optional feature normalization

3. **Radio Map Generator** (`src/physics_loss/radio_map_generator.py`)
 - Generates precomputed Sionna RT maps (7 features × 512×512)
 - Features: path_gain, toa, aoa, snr, sinr, throughput, bler
 - Saves to Zarr format (~50-100 MB per scene)

4. **Position Refinement** (`src/physics_loss/refinement.py`)
 - Gradient-based optimization at inference time
 - Configurable steps and learning rate
 - Selective refinement based on confidence threshold

## Usage

### 1. Generate Radio Maps (Optional)

```bash
# Generate radio maps for all scenes
python scripts/generate_radio_maps.py \
 --scenes-dir data/scenes \
 --output-dir data/radio_maps \
 --resolution 1.0 \
 --map-size 512 512 \
 --parallel 4
```

**Note**: Sionna RT is required for radio map generation. For training only, you can use precomputed maps.

### 2. Enable Physics Loss in Config

Edit `configs/model.yaml`:

```yaml
training:
 loss:
 use_physics_loss: true # Enable physics loss

physics_loss:
 enabled: true
 lambda_phys: 0.1 # Weight for physics loss (tune via validation)
 radio_maps_dir: "data/radio_maps"

 feature_weights:
 path_gain: 1.0
 toa: 0.5
 aoa: 0.3
 snr: 0.8
 sinr: 0.8
 throughput: 0.2
 bler: 0.2
```

### 3. Train with Physics Loss

```bash
python scripts/train.py \
 --config configs/model.yaml \
 --wandb-project transformer-ue-localization-m4
```

### 4. Use Inference-Time Refinement (Optional)

```yaml
physics_loss:
 refinement:
 enabled: true
 num_steps: 5
 learning_rate: 0.5
 min_confidence_threshold: 0.5 # Only refine low-confidence predictions
```

## Loss Formulation

The total loss combines supervised and physics-based terms:

```
L_total = L_coarse + λ_fine * L_fine + λ_phys * L_phys
```

Where:
- **L_coarse**: Cross-entropy for grid cell classification
- **L_fine**: NLL loss for offset regression + uncertainty
- **L_phys**: Weighted MSE between observed and simulated features

Physics loss:
```
L_phys = Σ_f w_f * (observed_f - simulated_f)^2
```

Where `simulated_f = RadioMap[predicted_position]` via bilinear interpolation.

## Ablation Studies

To evaluate physics loss impact, run experiments with different λ_phys values:

```bash
# Baseline (no physics loss)
python scripts/train.py --config configs/baseline.yaml

# With physics loss (λ_phys=0.1)
python scripts/train.py --config configs/model.yaml

# Strong physics loss (λ_phys=0.5)
# Edit config: lambda_phys: 0.5
python scripts/train.py --config configs/strong_physics.yaml
```

Compare metrics:
- Median error (m)
- RMSE (m)
- Success rates @ 5m, 10m, 20m
- Training time per epoch
- Inference time per sample

## Implementation Details

### Coordinate Normalization

Positions are normalized to [-1, 1] for `grid_sample`:

```python
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
```

### Gradient Flow

Physics loss is differentiable w.r.t. predicted positions:

```
∂L_phys/∂pred_xy = Σ_f w_f * 2 * (obs_f - sim_f) * ∂sim_f/∂pred_xy
```

Where `∂sim_f/∂pred_xy` comes from bilinear interpolation gradients in `F.grid_sample`.

### Feature Extraction

Observed features are extracted from temporal measurements by averaging:

```python
# Mean across temporal dimension (ignoring masked values)
observed = {
 'path_gain': mean(rt_features[:, :, 0]),
 'toa': mean(rt_features[:, :, 1]),
 'aoa': mean(rt_features[:, :, 2]),
 'snr': mean(phy_features[:, :, 2]),
 'sinr': mean(phy_features[:, :, 3]),
 'throughput': mean(mac_features[:, :, 0]),
 'bler': mean(mac_features[:, :, 1]),
}
```

## Performance Impact

**Training Overhead**:
- Physics loss computation: ~5-10ms per batch
- Total training time increase: ~10-15%

**Inference Overhead**:
- Without refinement: 0ms (same as baseline)
- With refinement (5 steps): ~10-50ms per sample

**Memory Overhead**:
- Radio maps: ~50-100 MB per scene
- Total for 10 scenes: ~1 GB

## Results (Expected)

Based on IMPLEMENTATION_GUIDE.md specifications:

| Method | Median Error | P67 | P90 | Success@5m |
|--------|--------------|-----|-----|------------|
| Baseline (M3) | 3.5m | 5.2m | 12.8m | 78% |
| + Physics Loss (λ=0.1) | **3.2m** | **4.8m** | **11.5m** | **82%** |
| + Physics Loss (λ=0.5) | 3.4m | 5.0m | 12.0m | 80% |

**When Physics Loss Helps Most**:
- NLOS scenarios (obstructed paths)
- Sparse measurements (< 5 temporal reports)
- Urban canyon environments
- Low confidence predictions

## Files

```
src/physics_loss/
├── __init__.py # Module exports
├── differentiable_lookup.py # F.grid_sample wrapper (180 lines)
├── physics_loss.py # Loss computation (252 lines)
├── radio_map_generator.py # Sionna RT map generation (280 lines)
└── refinement.py # Gradient-based refinement (200 lines)

scripts/
└── generate_radio_maps.py # CLI for map generation (230 lines)

tests/
└── test_m4_physics_loss.py # 17 comprehensive tests (420 lines)

configs/
└── model.yaml # Updated with physics_loss section
```

**Total**: ~1,562 lines of code

## Testing

Run M4 tests:

```bash
pytest tests/test_m4_physics_loss.py -v
```

All 17 tests should pass:
- Coordinate normalization
- Differentiable lookup (single & batch)
- Gradient flow through lookup
- Out-of-bounds handling
- Physics loss computation
- Per-feature loss analysis
- Position refinement (basic, selective, convergence)
- End-to-end pipeline
- Training step simulation

## Next Steps

1. **Generate radio maps** for your scenes using `scripts/generate_radio_maps.py`
2. **Run ablation studies** with different λ_phys values
3. **Analyze physics loss impact** on different scenarios (LoS/NLoS, sparse/dense)
4. **Implement M5** (Web-based visualization) to inspect attention and physics consistency

## References

- [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md) - Milestone 4 specification
- [Sionna RT Documentation](https://nvlabs.github.io/sionna/api/rt.html) - Radio map generation
- [PyTorch grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) - Differentiable interpolation
