"""
UE Localization Model: Main Integration

Integrates all components into end-to-end positioning system:
1. Radio Encoder: Process measurement sequences
2. Map Encoder: Process radio + OSM maps
3. Cross-Attention Fusion: Fuse radio and map features
4. Coarse Head: Predict grid cell probabilities
5. Fine Head: Refine position within top-K cells
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .radio_encoder import RadioEncoder
from .map_encoder import MapEncoder
from .fusion import CrossAttentionFusion
from .heads import CoarseHead, FineHead


class UELocalizationModel(nn.Module):
    """End-to-end UE localization model with coarse-to-fine prediction.
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        radio_cfg = config['model']['radio_encoder']
        map_cfg = config['model']['map_encoder']
        fusion_cfg = config['model']['fusion']
        coarse_cfg = config['model']['coarse_head']
        fine_cfg = config['model']['fine_head']
        dataset_cfg = config['dataset']
        
        # Store for later use
        self.grid_size = coarse_cfg['grid_size']
        scene_extent = dataset_cfg['scene_extent']
        
        # Handle scene_extent as either scalar or [x_min, y_min, x_max, y_max]
        if isinstance(scene_extent, (list, tuple)):
            # Extract size from extent
            x_min, y_min, x_max, y_max = scene_extent
            self.scene_extent = x_max - x_min  # Assume square scenes
            self.map_extent = tuple(scene_extent)
        else:
            # Scalar size (legacy format)
            self.scene_extent = scene_extent
            self.map_extent = (0.0, 0.0, scene_extent, scene_extent)
        
        self.cell_size = self.scene_extent / self.grid_size
        self.top_k = fine_cfg['top_k']
        
        # Build components
        self.radio_encoder = RadioEncoder(
            num_cells=radio_cfg['num_cells'],
            num_beams=radio_cfg['num_beams'],
            d_model=radio_cfg['d_model'],
            nhead=radio_cfg['nhead'],
            num_layers=radio_cfg['num_layers'],
            dropout=radio_cfg['dropout'],
            max_seq_len=radio_cfg['max_seq_len'],
            rt_features_dim=radio_cfg['rt_features_dim'],
            phy_features_dim=radio_cfg['phy_features_dim'],
            mac_features_dim=radio_cfg['mac_features_dim'],
        )
        
        self.map_encoder = MapEncoder(
            img_size=map_cfg['img_size'],
            patch_size=map_cfg['patch_size'],
            in_channels=map_cfg['in_channels'],
            d_model=map_cfg['d_model'],
            nhead=map_cfg['nhead'],
            num_layers=map_cfg['num_layers'],
            dropout=map_cfg['dropout'],
            radio_map_channels=map_cfg['radio_map_channels'],
            osm_map_channels=map_cfg['osm_map_channels'],
        )
        
        self.fusion = CrossAttentionFusion(
            d_radio=radio_cfg['d_model'],
            d_map=map_cfg['d_model'],
            d_fusion=fusion_cfg['d_fusion'],
            nhead=fusion_cfg['nhead'],
            dropout=fusion_cfg['dropout'],
        )
        
        self.coarse_head = CoarseHead(
            d_input=fusion_cfg['d_fusion'],
            grid_size=coarse_cfg['grid_size'],
            dropout=coarse_cfg['dropout'],
        )
        
        self.fine_head = FineHead(
            d_input=fusion_cfg['d_fusion'],
            d_hidden=fine_cfg['d_hidden'],
            top_k=fine_cfg['top_k'],
            dropout=fine_cfg['dropout'],
        )
    
    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            measurements: Dict with keys:
                - rt_features: [batch, seq_len, rt_dim]
                - phy_features: [batch, seq_len, phy_dim]
                - mac_features: [batch, seq_len, mac_dim]
                - cell_ids: [batch, seq_len]
                - beam_ids: [batch, seq_len]
                - timestamps: [batch, seq_len]
                - mask: [batch, seq_len]
            radio_map: [batch, 5, H, W]
            osm_map: [batch, 5, H, W]
        
        Returns:
            Dictionary with predictions:
                - coarse_logits: [batch, num_cells]
                - coarse_heatmap: [batch, grid_size, grid_size]
                - top_k_indices: [batch, k]
                - top_k_probs: [batch, k]
                - fine_offsets: [batch, k, 2]
                - fine_uncertainties: [batch, k, 2]
                - predicted_position: [batch, 2] (best prediction)
        """
        # 1. Encode radio measurements
        radio_emb = self.radio_encoder(measurements)  # [B, d_radio]
        
        # 2. Encode maps
        map_tokens, map_cls = self.map_encoder(radio_map, osm_map)  # [B, N, d_map], [B, d_map]
        
        # 3. Fuse radio and map features
        fused = self.fusion(radio_emb, map_tokens)  # [B, d_fusion]
        
        # 4. Coarse prediction
        coarse_logits, coarse_heatmap = self.coarse_head(fused)  # [B, num_cells], [B, H, W]
        
        # 5. Get top-K cells
        top_k_indices, top_k_probs = self.coarse_head.get_top_k_cells(
            coarse_heatmap,
            k=self.top_k,
        )  # [B, k], [B, k]
        
        # 6. Fine refinement for top-K cells
        fine_offsets, fine_uncertainties = self.fine_head(
            fused,
            top_k_indices,
        )  # [B, k, 2], [B, k, 2]
        
        # 7. Convert to final position prediction (use highest probability cell)
        top_cell_coords = self.coarse_head.indices_to_coords(
            top_k_indices[:, 0:1],  # Take best cell
            self.cell_size,
        )  # [B, 1, 2]
        
        predicted_position = top_cell_coords.squeeze(1) + fine_offsets[:, 0, :]  # [B, 2]
        
        return {
            'coarse_logits': coarse_logits,
            'coarse_heatmap': coarse_heatmap,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'fine_offsets': fine_offsets,
            'fine_uncertainties': fine_uncertainties,
            'predicted_position': predicted_position,
        }
    
    def predict(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Simplified inference interface.
        
        Args:
            measurements: Measurement dictionary
            radio_map: Radio map tensor
            osm_map: OSM map tensor
            return_uncertainty: Whether to return uncertainty estimate
        
        Returns:
            predicted_position: [batch, 2] (x, y) in meters
            uncertainty: [batch, 2] (σx, σy) in meters (if return_uncertainty=True)
        """
        outputs = self.forward(measurements, radio_map, osm_map)
        
        position = outputs['predicted_position']
        
        if return_uncertainty:
            # Return uncertainty of best prediction
            uncertainty = outputs['fine_uncertainties'][:, 0, :]  # [B, 2]
            return position, uncertainty
        
        return position, None
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss and components.
        
        Args:
            outputs: Model outputs from forward()
            targets: Ground truth dictionary with:
                - position: [batch, 2] true (x, y)
                - cell_grid: [batch] true cell index
            loss_weights: Dict with 'coarse_weight' and 'fine_weight'
        
        Returns:
            Dictionary with:
                - loss: Total loss
                - coarse_loss: Cross-entropy loss
                - fine_loss: NLL loss for offset
        """
        # Coarse loss: cross-entropy on grid cells
        coarse_loss = nn.functional.cross_entropy(
            outputs['coarse_logits'],
            targets['cell_grid'],
        )
        
        # Fine loss: NLL for offset prediction
        # Compute true offset from best predicted cell
        top_cell_coords = self.coarse_head.indices_to_coords(
            outputs['top_k_indices'][:, 0:1],
            self.cell_size,
        ).squeeze(1)  # [B, 2]
        
        true_offset = targets['position'] - top_cell_coords  # [B, 2]
        
        fine_loss = self.fine_head.nll_loss(
            outputs['fine_offsets'],
            outputs['fine_uncertainties'],
            true_offset,
            outputs['top_k_probs'],
        )
        
        # Total loss
        total_loss = (
            loss_weights['coarse_weight'] * coarse_loss +
            loss_weights['fine_weight'] * fine_loss
        )
        
        return {
            'loss': total_loss,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss,
        }
    
    def forward_with_attention(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with attention weights for visualization.
        
        Returns:
            outputs: Regular model outputs
            attention_weights: [batch, 1, num_patches] cross-attention weights
        """
        # Encode
        radio_emb = self.radio_encoder(measurements)
        map_tokens, map_cls = self.map_encoder(radio_map, osm_map)
        
        # Fuse with attention tracking
        fused, attention_weights = self.fusion.forward_with_attention(
            radio_emb,
            map_tokens,
        )
        
        # Rest of forward pass
        coarse_logits, coarse_heatmap = self.coarse_head(fused)
        top_k_indices, top_k_probs = self.coarse_head.get_top_k_cells(
            coarse_heatmap,
            k=self.top_k,
        )
        fine_offsets, fine_uncertainties = self.fine_head(fused, top_k_indices)
        
        top_cell_coords = self.coarse_head.indices_to_coords(
            top_k_indices[:, 0:1],
            self.cell_size,
        ).squeeze(1)
        predicted_position = top_cell_coords + fine_offsets[:, 0, :]
        
        outputs = {
            'coarse_logits': coarse_logits,
            'coarse_heatmap': coarse_heatmap,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'fine_offsets': fine_offsets,
            'fine_uncertainties': fine_uncertainties,
            'predicted_position': predicted_position,
        }
        
        return outputs, attention_weights
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
