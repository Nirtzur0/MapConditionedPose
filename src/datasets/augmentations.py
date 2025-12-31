"""
Data Augmentations for Radio Localization Training

Provides runtime augmentations for measurements and maps to improve
model generalization. All augmentations are applied only during training.
"""

import torch
from typing import Dict, Optional, Tuple


class RadioAugmentation(torch.nn.Module):
    """Applies augmentations to radio measurements and maps on GPU.
    
    Supports both single-sample and batched inputs.
    
    Args:
        config: Augmentation configuration dict with keys:
            - feature_noise: Gaussian noise std for features
            - feature_dropout: Probability of zeroing features
            - temporal_dropout: Probability of dropping time steps
            - random_flip: Enable random horizontal flip
            - random_rotation: Enable random 90° rotations
            - scale_range: [min, max] for random scale jitter
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.enabled = bool(self.config)
    
    def __call__(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply all augmentations.
        
        Returns:
            Tuple of (augmented_measurements, augmented_radio_map, augmented_osm_map, augmented_position)
        """
        if not self.enabled:
            return measurements, radio_map, osm_map, position
        
        measurements = self.augment_measurements(measurements)
        radio_map, osm_map, position = self.augment_maps(radio_map, osm_map, position)
        
        return measurements, radio_map, osm_map, position
    
    def augment_measurements(self, measurements: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentations to measurement features.
        
        Augmentations:
        - feature_noise: Add Gaussian noise to RT/PHY/MAC features
        - temporal_dropout: Randomly drop time steps
        - feature_dropout: Randomly zero out individual features
        """
        cfg = self.config
        
        # Feature noise: Add Gaussian jitter to measurements
        if cfg.get('feature_noise', 0.0) > 0:
            noise_std = cfg['feature_noise']
            
            # RT features: Add proportional noise
            rt_feat = measurements['rt_features']
            rt_noise = torch.randn_like(rt_feat) * noise_std * (torch.abs(rt_feat) + 1e-6)
            measurements['rt_features'] = rt_feat + rt_noise
            
            # PHY features: Add absolute noise (different scales for different features)
            phy_feat = measurements['phy_features']
            # RSRP/RSRQ/SINR: ±2-5 dB noise, CQI: ±1, RI/PMI: no noise
            phy_noise_scales = torch.tensor([3.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 0.0])[:phy_feat.shape[-1]]
            phy_noise = torch.randn_like(phy_feat) * noise_std * phy_noise_scales
            measurements['phy_features'] = phy_feat + phy_noise
            
            # MAC features: Proportional noise
            mac_feat = measurements['mac_features']
            mac_noise = torch.randn_like(mac_feat) * noise_std * 0.5
            measurements['mac_features'] = mac_feat + mac_noise
        
        # Temporal dropout: Randomly mask some time steps
        if cfg.get('temporal_dropout', 0.0) > 0:
            drop_prob = cfg['temporal_dropout']
            mask = measurements['mask']
            # Only drop if we have multiple time steps
            if mask.sum() > 1:
                drop_mask = torch.rand(mask.shape) > drop_prob
                # Ensure at least one time step remains
                if drop_mask.any():
                    measurements['mask'] = mask & drop_mask
        
        # Feature dropout: Zero out random individual features
        if cfg.get('feature_dropout', 0.0) > 0:
            drop_prob = cfg['feature_dropout']
            
            # Drop RT features
            rt_drop = torch.rand(measurements['rt_features'].shape) < drop_prob
            measurements['rt_features'] = measurements['rt_features'].masked_fill(rt_drop, 0.0)
            
            # Drop PHY features (per-feature, not per-timestep)
            phy_drop = torch.rand(measurements['phy_features'].shape[-1]) < drop_prob
            measurements['phy_features'][..., phy_drop] = 0.0
        
        return measurements
    
    def augment_maps(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply geometric augmentations to maps and position.
        
        Augmentations:
        - rotation: Random rotation (applied to both maps consistently)
        - flip: Random horizontal flip
        - scale: Random scale jitter
        """
        cfg = self.config
        device = radio_map.device
        
        # Determine if batch or single
        is_batch = radio_map.dim() == 4
        
        # Ensure position is at least 2D [B, 2] or [1, 2] if single
        if position is not None:
             if not is_batch and position.dim() == 1:
                  position = position.unsqueeze(0) # [1, 2]
             
        # Random horizontal flip (50% chance)
        if cfg.get('random_flip', True):
            # Generate random mask
            if is_batch:
                do_flip = torch.rand(radio_map.shape[0], device=device) > 0.5
                # Maps: [B, C, H, W] -> flip last dim
                # Position: [B, 2] -> x = 1.0 - x
                
                # Expand dims for maps
                flip_mask_map = do_flip.view(-1, 1, 1, 1)
                
                # Apply flip to maps (where mask is True)
                radio_map_flipped = torch.flip(radio_map, dims=[-1])
                radio_map = torch.where(flip_mask_map, radio_map_flipped, radio_map)
                
                osm_map_flipped = torch.flip(osm_map, dims=[-1])
                osm_map = torch.where(flip_mask_map, osm_map_flipped, osm_map)
                
                # Apply flip to position (where mask is True)
                if position is not None:
                    # x_new = 1.0 - x
                    pos_flipped = position.clone()
                    pos_flipped[:, 0] = 1.0 - pos_flipped[:, 0]
                    
                    flip_mask_pos = do_flip.unsqueeze(1)
                    position = torch.where(flip_mask_pos, pos_flipped, position)
            else:
                if torch.rand(1, device=device).item() > 0.5:
                    radio_map = torch.flip(radio_map, dims=[-1])
                    osm_map = torch.flip(osm_map, dims=[-1])
                    if position is not None:
                         position[:, 0] = 1.0 - position[:, 0]
        
        # Random 90-degree rotations (0, 90, 180, 270)
        # Note: torch.rot90 rotates CCW
        # k=1 (90): (x, y) -> (1-y, x)
        # k=2 (180): (x, y) -> (1-x, 1-y)
        # k=3 (270): (x, y) -> (y, 1-x)
        if cfg.get('random_rotation', True):
            # For simplicity in batch, pick ONE rotation for the whole batch or per-sample?
            # Per-sample is better but complex with rot90.
            # Let's do per-sample by iterating or using affine_grid (expensive).
            # Optimization: Just pick one K for the whole batch for now (much faster)
            # Or group by K. 
            
            # Simple approach: same rotation for whole batch (efficient)
            k = torch.randint(0, 4, (1,), device=device).item()
            if k > 0:
                radio_map = torch.rot90(radio_map, k, dims=[-2, -1])
                osm_map = torch.rot90(osm_map, k, dims=[-2, -1])
                
                if position is not None:
                    x = position[:, 0].clone()
                    y = position[:, 1].clone()
                    
                    if k == 1: # 90 CCW
                        position[:, 0] = 1.0 - y
                        position[:, 1] = x
                    elif k == 2: # 180
                        position[:, 0] = 1.0 - x
                        position[:, 1] = 1.0 - y
                    elif k == 3: # 270 CCW
                        position[:, 0] = y
                        position[:, 1] = 1.0 - x
        
        # Scale jitter: Randomly zoom in/out and crop back
        # Note: Keeping simple batch-wide scale
        scale_range = cfg.get('scale_range', None)
        if scale_range is not None and len(scale_range) == 2:
            scale = scale_range[0] + torch.rand(1, device=device).item() * (scale_range[1] - scale_range[0])
            
            if abs(scale - 1.0) > 0.01:
                # Get dimensions
                if radio_map.dim() == 4:
                    H, W = radio_map.shape[-2:]
                else:
                    H, W = radio_map.shape[-2:]
                
                new_H, new_W = int(H * scale), int(W * scale)
                
                # Interpolate (Resize)
                # Input to interpolate must be 4D [B, C, H, W]
                rm_input = radio_map if is_batch else radio_map.unsqueeze(0)
                om_input = osm_map if is_batch else osm_map.unsqueeze(0)
                
                radio_map = torch.nn.functional.interpolate(
                    rm_input, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                osm_map = torch.nn.functional.interpolate(
                    om_input, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                
                # Position Update for Scale (Zoom/Shrink)
                # "Zoom implies field of view shrinks" -> Points move apart
                # Formula: x_new = (x - 0.5) * scale + 0.5
                if position is not None:
                     position = (position - 0.5) * scale + 0.5
                
                # Center crop/pad
                if new_H >= H and new_W >= W:
                    # Crop center
                    start_h = (new_H - H) // 2
                    start_w = (new_W - W) // 2
                    radio_map = radio_map[..., start_h:start_h+H, start_w:start_w+W]
                    osm_map = osm_map[..., start_h:start_h+H, start_w:start_w+W]
                    
                    # If we zoomed in (scale > 1), we effectively cropped.
                    # Position update is correct (points moved away from center).
                    # Points outside [0, 1] are now off-screen.
                else:
                    # Scale down: Resize back to H,W to avoid padding artifacts?
                    # Or pad with zeros. Padding with zeros is safer for maps.
                    # Create canvas
                    rm_canvas = torch.zeros_like(rm_input)
                    om_canvas = torch.zeros_like(om_input)
                    
                    offset_h = (H - new_H) // 2
                    offset_w = (W - new_W) // 2
                    
                    rm_canvas[..., offset_h:offset_h+new_H, offset_w:offset_w+new_W] = radio_map
                    om_canvas[..., offset_h:offset_h+new_H, offset_w:offset_w+new_W] = osm_map
                    
                    radio_map = rm_canvas
                    osm_map = om_canvas
                
                if not is_batch:
                    radio_map = radio_map.squeeze(0)
                    osm_map = osm_map.squeeze(0)
                    if position is not None:
                        position = position.squeeze(0)

        return radio_map, osm_map, position
