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
from .map_encoder import E2EquivariantMapEncoder, StandardMapEncoder
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
        # Normalize coordinate system to [0, 1] for scale invariance
        # The model predicts normalized coordinates relative to the scene size.
        # This handles variable scene sizes (500m vs 2000m) seamlessly.
        self.scene_extent = 1.0
        self.map_extent = (0.0, 0.0, 1.0, 1.0)
        self.origin = (0.0, 0.0)
        
        self.cell_size = self.scene_extent / self.grid_size
        self.top_k = fine_cfg['top_k']
        self.use_local_map = fine_cfg.get('use_local_map', False)
        self.fine_patch_size = fine_cfg.get('patch_size', 64)
        
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
            cfr_enabled=radio_cfg.get('cfr_enabled', False),
            cfr_num_cells=radio_cfg.get('cfr_num_cells', 8),
            cfr_num_subcarriers=radio_cfg.get('cfr_num_subcarriers', 64),
        )
        
        # Map encoder: Choose between E2 equivariant and standard ViT
        use_e2 = map_cfg.get('use_e2_equivariant', False)
        if use_e2:
            self.map_encoder = E2EquivariantMapEncoder(
                img_size=map_cfg['img_size'],
                patch_size=map_cfg.get('patch_size', 16),  # Pass patch_size!
                in_channels=map_cfg['in_channels'],
                d_model=map_cfg['d_model'],
                num_heads=map_cfg['nhead'],
                num_layers=map_cfg['num_layers'],
                num_group_elements=map_cfg.get('num_group_elements', 8),  # p4m group by default
                dropout=map_cfg['dropout'],
                radio_map_channels=map_cfg['radio_map_channels'],
                osm_map_channels=map_cfg['osm_map_channels'],
                cache_size=map_cfg.get('cache_size', 0),
                cache_mode=map_cfg.get('cache_mode', 'off'),
            )
        else:
            self.map_encoder = StandardMapEncoder(
                img_size=map_cfg['img_size'],
                patch_size=map_cfg.get('patch_size', 16),
                in_channels=map_cfg['in_channels'],
                d_model=map_cfg['d_model'],
                num_heads=map_cfg['nhead'],
                num_layers=map_cfg['num_layers'],
                dropout=map_cfg['dropout'],
                radio_map_channels=map_cfg['radio_map_channels'],
                osm_map_channels=map_cfg['osm_map_channels'],
                cache_size=map_cfg.get('cache_size', 0),
                cache_mode=map_cfg.get('cache_mode', 'off'),
            )
        
        self.fusion = CrossAttentionFusion(
            d_radio=radio_cfg['d_model'],
            d_map=map_cfg['d_model'],
            d_fusion=fusion_cfg['d_fusion'],
            nhead=fusion_cfg['nhead'],
            dropout=fusion_cfg['dropout'],
            num_query_tokens=fusion_cfg.get('num_query_tokens', 4),  # Multi-query attention
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
            num_cells=self.grid_size ** 2,
            d_map=map_cfg['d_model'],
            use_local_map=self.use_local_map,
            offset_scale=fine_cfg.get('offset_scale', 1.5),
            sigma_min_ratio=fine_cfg.get('sigma_min_ratio', 0.03),
            sigma_max_ratio=fine_cfg.get('sigma_max_ratio', 3.2),
            dropout=fine_cfg['dropout'],
        )

        aux_cfg = config.get('training', {}).get('loss', {}).get('auxiliary', {})
        self.aux_enabled = aux_cfg.get('enabled', False)
        self.aux_input = aux_cfg.get('input', 'radio')
        self.aux_tasks = aux_cfg.get('tasks', {}) or {}
        self.aux_heads = nn.ModuleDict()
        if self.aux_enabled and self.aux_tasks:
            aux_dim = radio_cfg['d_model'] if self.aux_input == 'radio' else fusion_cfg['d_fusion']
            hidden_dim = aux_cfg.get('hidden_dim') or aux_dim
            for task_name in self.aux_tasks.keys():
                self.aux_heads[task_name] = nn.Sequential(
                    nn.Linear(aux_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(fine_cfg['dropout']),
                    nn.Linear(hidden_dim, 1),
                )

    def _indices_to_pixel_centers(
        self,
        indices: torch.Tensor,
        map_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Convert grid indices to pixel centers in the map tensor."""
        height = map_tensor.shape[2]
        width = map_tensor.shape[3]
        rows = indices // self.grid_size
        cols = indices % self.grid_size
        cell_w = width / self.grid_size
        cell_h = height / self.grid_size
        centers_x = (cols.float() + 0.5) * cell_w
        centers_y = (rows.float() + 0.5) * cell_h
        return torch.stack([centers_x, centers_y], dim=-1)
    
    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        scene_idx: Optional[torch.Tensor] = None,
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
            scene_idx: Optional scene indices for map-encoder caching
        
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
        map_tokens, map_cls = self.map_encoder(
            radio_map,
            osm_map,
            cache_keys=scene_idx,
        )  # [B, N, d_map], [B, d_map]
        
        # 3. Fuse radio and map features
        fused = self.fusion(radio_emb, map_tokens)  # [B, d_fusion]

        aux_outputs = None
        if self.aux_enabled and self.aux_heads:
            aux_input = radio_emb if self.aux_input == 'radio' else fused
            aux_outputs = {
                name: head(aux_input).squeeze(-1)
                for name, head in self.aux_heads.items()
            }
        
        # 4. Coarse prediction
        coarse_logits, coarse_heatmap = self.coarse_head(fused)  # [B, num_cells], [B, H, W]
        
        # 5. Get top-K cells
        top_k_indices, top_k_probs = self.coarse_head.get_top_k_cells(
            coarse_heatmap,
            k=self.top_k,
        )  # [B, k], [B, k]

        # 6. Fine refinement for top-K cells
        local_map_embeddings = None
        if self.use_local_map:
            centers_px = self._indices_to_pixel_centers(top_k_indices, radio_map)
            local_map_embeddings = self.map_encoder.encode_local_patches(
                radio_map,
                osm_map,
                centers_px=centers_px,
                patch_size=self.fine_patch_size,
            )
        fine_offsets, fine_uncertainties = self.fine_head(
            fused,
            top_k_indices,
            cell_size=self.cell_size,
            local_map_embeddings=local_map_embeddings,
        )  # [B, k, 2], [B, k, 2]
        
        # 7. Convert to final position prediction (use highest probability cell)
        top_cell_coords = self.coarse_head.indices_to_coords(
            top_k_indices[:, 0:1],  # Take best cell
            self.cell_size,
            origin=self.origin,
        )  # [B, 1, 2]
        
        predicted_position = top_cell_coords.squeeze(1) + fine_offsets[:, 0, :]  # [B, 2]
        predicted_position = torch.clamp(predicted_position, 0.0, 1.0 - 1e-6)
        
        return {
            'coarse_logits': coarse_logits,
            'coarse_heatmap': coarse_heatmap,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'fine_offsets': fine_offsets,
            'fine_uncertainties': fine_uncertainties,
            'predicted_position': predicted_position,
            'aux_outputs': aux_outputs,
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
                - fine_loss: Mixture NLL loss
        """
        # Coarse loss: cross-entropy on grid cells
        coarse_loss = nn.functional.cross_entropy(
            outputs['coarse_logits'],
            targets['cell_grid'],
        )
        
        # Fine loss: Mixture NLL
        # 1. Compute centers for all top-K candidates
        top_k_centers = self.coarse_head.indices_to_coords(
            outputs['top_k_indices'],
            self.cell_size,
            origin=self.origin,
        )  # [B, K, 2]
        
        # 2. Compute targets relative to each candidate center
        # target_pos: [B, 1, 2], centers: [B, K, 2] -> offsets: [B, K, 2]
        target_offsets = targets['position'].unsqueeze(1) - top_k_centers
        
        # 3. Compute log probability for each component k
        # log_pi: weights from coarse head
        # Renormalize top-k probabilities to sum to 1 across the K candidates
        eps = 1e-6
        top_k_probs = outputs['top_k_probs']
        pi = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + eps)
        log_pi = torch.log(pi + eps)
        
        # log_N: Gaussian density
        # Model predicts offset (mu) and scale (sigma) from the cell center
        mu = outputs['fine_offsets']           # [B, K, 2]
        sigma = outputs['fine_uncertainties']  # [B, K, 2]
        
        # Residual between true offset and predicted offset
        residuals = target_offsets - mu        # [B, K, 2]
        var = sigma ** 2
        
        # Log likelihood per dimension: -0.5 * log(2πσ²) - 0.5 * (x-μ)²/σ²
        log_prob_dim = -0.5 * torch.log(2 * torch.pi * var + eps) - 0.5 * (residuals ** 2) / (var + eps)
        log_prob_xy = log_prob_dim.sum(dim=-1)  # [B, K] sum over x,y
        
        # Clip log probabilities to prevent numerical issues when coarse predictions are very wrong
        # Minimum corresponds to ~3 sigma away (log(0.01) ≈ -4.6)
        log_prob_xy = torch.clamp(log_prob_xy, min=-15.0)
        
        # 4. Total mixture probability: sum_k (pi_k * N_k)
        # In log domain: logsumexp(log_pi + log_N)
        log_mixture = torch.logsumexp(log_pi + log_prob_xy, dim=-1)  # [B]
        
        # Ensure numerical stability
        log_mixture = torch.clamp(log_mixture, min=-15.0)
        
        fine_loss = -log_mixture.mean()
        
        # 5. Auxiliary position loss (direct distance supervision)
        # This helps gradient flow and provides direct signal on position error
        pred_pos = outputs['predicted_position']  # [B, 2]
        true_pos = targets['position']  # [B, 2]
        position_loss = nn.functional.smooth_l1_loss(pred_pos, true_pos)
        
        # 6. Variance regularization loss - penalize large variances to encourage confident predictions
        # This encourages the model to predict smaller, more confident ellipses
        # We regularize log(sigma) to encourage smaller sigma values, weighted by mixture probability
        variance_reg_weight = loss_weights.get('variance_reg_weight', 0.1)
        # Weight each component's variance by its probability (focus on top predictions)
        weighted_log_sigma = (pi.detach() * torch.log(sigma + eps)).sum(dim=1)  # [B, 2]
        variance_loss = weighted_log_sigma.mean()  # Scalar, penalizes large sigma
        
        # Get position loss weight (default to 0.2 if not specified)
        position_weight = loss_weights.get('position_weight', 0.2)
        
        # Total loss
        total_loss = (
            loss_weights['coarse_weight'] * coarse_loss +
            loss_weights['fine_weight'] * fine_loss +
            position_weight * position_loss +
            variance_reg_weight * variance_loss
        )
        
        return {
            'loss': total_loss,
            'coarse_loss': coarse_loss,
            'fine_loss': fine_loss,
            'position_loss': position_loss,
            'variance_loss': variance_loss,
        }
    
    def forward_with_attention(
        self,
        measurements: Dict[str, torch.Tensor],
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        scene_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass with attention weights for visualization.
        
        Returns:
            outputs: Regular model outputs
            attention_weights: [batch, 1, num_patches] cross-attention weights
        """
        # Encode
        radio_emb = self.radio_encoder(measurements)
        map_tokens, map_cls = self.map_encoder(
            radio_map,
            osm_map,
            cache_keys=scene_idx,
        )
        
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
        local_map_embeddings = None
        if self.use_local_map:
            centers_px = self._indices_to_pixel_centers(top_k_indices, radio_map)
            local_map_embeddings = self.map_encoder.encode_local_patches(
                radio_map,
                osm_map,
                centers_px=centers_px,
                patch_size=self.fine_patch_size,
            )
        fine_offsets, fine_uncertainties = self.fine_head(
            fused,
            top_k_indices,
            cell_size=self.cell_size,
            local_map_embeddings=local_map_embeddings,
        )
        
        top_cell_coords = self.coarse_head.indices_to_coords(
            top_k_indices[:, 0:1],
            self.cell_size,
            origin=self.origin,
        ).squeeze(1)
        predicted_position = top_cell_coords + fine_offsets[:, 0, :]
        predicted_position = torch.clamp(predicted_position, 0.0, 1.0 - 1e-6)

        aux_outputs = None
        if self.aux_enabled and self.aux_heads:
            aux_input = radio_emb if self.aux_input == 'radio' else fused
            aux_outputs = {
                name: head(aux_input).squeeze(-1)
                for name, head in self.aux_heads.items()
            }

        outputs = {
            'coarse_logits': coarse_logits,
            'coarse_heatmap': coarse_heatmap,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'fine_offsets': fine_offsets,
            'fine_uncertainties': fine_uncertainties,
            'predicted_position': predicted_position,
            'aux_outputs': aux_outputs,
        }
        
        return outputs, attention_weights
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
