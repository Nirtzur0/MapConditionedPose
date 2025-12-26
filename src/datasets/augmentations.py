"""
Data Augmentations for Radio Localization Training

Provides runtime augmentations for measurements and maps to improve
model generalization. All augmentations are applied only during training.
"""

import torch
from typing import Dict, Optional, Tuple


class RadioAugmentation:
    """Applies augmentations to radio measurements and maps.
    
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
        self.config = config or {}
        self.enabled = bool(self.config)
    
    def __call__(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply all augmentations.
        
        Returns:
            Tuple of (augmented_measurements, augmented_radio_map, augmented_osm_map)
        """
        if not self.enabled:
            return measurements, radio_map, osm_map
        
        measurements = self.augment_measurements(measurements)
        radio_map, osm_map = self.augment_maps(radio_map, osm_map)
        
        return measurements, radio_map, osm_map
    
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geometric augmentations to maps.
        
        Augmentations:
        - rotation: Random rotation (applied to both maps consistently)
        - flip: Random horizontal flip
        - scale: Random scale jitter
        """
        cfg = self.config
        
        # Random horizontal flip (50% chance)
        if cfg.get('random_flip', True) and torch.rand(1).item() > 0.5:
            radio_map = torch.flip(radio_map, dims=[-1])
            osm_map = torch.flip(osm_map, dims=[-1])
        
        # Random 90-degree rotations (0, 90, 180, 270)
        if cfg.get('random_rotation', True):
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                radio_map = torch.rot90(radio_map, k, dims=[-2, -1])
                osm_map = torch.rot90(osm_map, k, dims=[-2, -1])
        
        # Scale jitter: Randomly zoom in/out and crop back
        scale_range = cfg.get('scale_range', None)
        if scale_range is not None and len(scale_range) == 2:
            scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
            if abs(scale - 1.0) > 0.01:  # Only if meaningful scale change
                H, W = radio_map.shape[-2:]
                new_H, new_W = int(H * scale), int(W * scale)
                
                # Scale up
                radio_map = torch.nn.functional.interpolate(
                    radio_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
                ).squeeze(0)
                osm_map = torch.nn.functional.interpolate(
                    osm_map.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False
                ).squeeze(0)
                
                # Center crop back to original size
                if new_H >= H and new_W >= W:
                    start_h = (new_H - H) // 2
                    start_w = (new_W - W) // 2
                    radio_map = radio_map[:, start_h:start_h+H, start_w:start_w+W]
                    osm_map = osm_map[:, start_h:start_h+H, start_w:start_w+W]
                else:
                    # Scale down: resize back
                    radio_map = torch.nn.functional.interpolate(
                        radio_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)
                    osm_map = torch.nn.functional.interpolate(
                        osm_map.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)
        
        return radio_map, osm_map
