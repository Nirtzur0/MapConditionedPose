"""
Radio Encoder: Temporal Set Transformer

Processes sparse temporal measurement sequences with:
- Embeddings for cell_id, beam_id, time
- Feature projection for RT/PHY/MAC measurements
- Transformer encoder with masked attention
- CLS token for sequence representation
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestamps.
    
    Uses continuous encoding based on relative time from start of sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, time_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.time_scale = time_scale
        
        # Precompute division term for sinusoidal encoding
        # exp(arange(0, d, 2) * (-log(10000)/d))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [batch, seq_len] continuous time values
        Returns:
            Positional encodings [batch, seq_len, d_model]
        """
        # Compute relative time from start of sequence
        # timestamps: [B, L]
        if timestamps.size(1) > 0:
            t_rel = timestamps - timestamps[:, :1]
        else:
            t_rel = timestamps
            
        # Scale time
        t = t_rel / self.time_scale
        
        # Compute phase: [B, L, 1] * [d_model/2] -> [B, L, d_model/2]
        phase = t.unsqueeze(-1) * self.div_term
        
        # Compute sin/cos
        batch_size, seq_len = timestamps.shape
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=timestamps.device)
        pe[..., 0::2] = torch.sin(phase)
        pe[..., 1::2] = torch.cos(phase)
        
        return pe


class RadioEncoder(nn.Module):
    """Encoder for sparse temporal measurement sequences.
    
    Architecture:
        1. Embeddings: cell_id, beam_id (learned) + time (sinusoidal)
        2. Feature projection: RT/PHY/MAC features -> d_model
        3. Transformer encoder: self-attention with masking
        4. CLS token: aggregates sequence information
    
    Args:
        num_cells: Maximum number of unique cell IDs
        num_beams: Maximum number of beam IDs
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        rt_features_dim: Dimension of RT layer features
        phy_features_dim: Dimension of PHY layer features
        mac_features_dim: Dimension of MAC layer features
    """
    
    def __init__(
        self,
        num_cells: int = 512,
        num_beams: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 20,
        rt_features_dim: int = 8,
        phy_features_dim: int = 10,
        mac_features_dim: int = 6,
        time_scale: float = 1.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_cells = num_cells
        self.num_beams = num_beams
        
        # Embeddings
        self.cell_embedding = nn.Embedding(num_cells, d_model // 4)
        self.beam_embedding = nn.Embedding(num_beams, d_model // 4)
        self.pos_encoding = PositionalEncoding(d_model // 4, max_len=max_seq_len * 10, time_scale=time_scale)
        
        # Feature projections with LayerNorm for stable training
        self.rt_projection = nn.Sequential(
            nn.Linear(rt_features_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
        )
        self.phy_projection = nn.Sequential(
            nn.Linear(phy_features_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
        )
        self.mac_projection = nn.Sequential(
            nn.Linear(mac_features_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
        )
        
        # Combine embeddings and features
        # Total: cell_emb + beam_emb + pos_enc + rt + phy + mac + ... -> d_model
        self.input_projection = nn.Linear(d_model // 4 * 6, d_model)
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        measurements: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            measurements: Dictionary containing:
                - rt_features: [batch, seq_len, rt_dim]
                - phy_features: [batch, seq_len, phy_dim]
                - mac_features: [batch, seq_len, mac_dim]
                - cell_ids: [batch, seq_len]
                - beam_ids: [batch, seq_len]
                - timestamps: [batch, seq_len]
                - mask: [batch, seq_len] (True = valid, False = padding/missing)
        
        Returns:
            CLS token embedding [batch, d_model]
        """
        batch_size, seq_len = measurements['cell_ids'].shape
        device = measurements['cell_ids'].device
        
        # 1. Embeddings
        cell_emb = self.cell_embedding(measurements['cell_ids'])  # [B, L, d/4]
        beam_emb = self.beam_embedding(measurements['beam_ids'])  # [B, L, d/4]
        time_emb = self.pos_encoding(measurements['timestamps'])  # [B, L, d/4]
        
        # 2. Feature projections
        rt_proj = self.rt_projection(measurements['rt_features'])  # [B, L, d/4]
        phy_proj = self.phy_projection(measurements['phy_features'])  # [B, L, d/4]
        mac_proj = self.mac_projection(measurements['mac_features'])  # [B, L, d/4]
        
        # 3. Concatenate all components
        combined = torch.cat([
            cell_emb,
            beam_emb,
            time_emb,
            rt_proj,
            phy_proj,
            mac_proj,
        ], dim=-1)  # [B, L, d/4 * 6]
        
        # 4. Project to d_model
        tokens = self.input_projection(combined)  # [B, L, d_model]
        tokens = self.dropout(tokens)
        
        # 5. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, L+1, d_model]
        
        # 6. Create attention mask for transformer
        # Extend mask for CLS token (always valid)
        mask = measurements['mask']  # [B, L]
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)  # [B, L+1]
        
        # Transformer expects inverted mask (True = ignore, False = attend)
        # But also needs key_padding_mask format: True = padding, False = valid
        src_key_padding_mask = ~extended_mask  # [B, L+1]
        
        # 7. Transformer encoding
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, L+1, d_model]
        
        # 8. Extract CLS token
        cls_output = encoded[:, 0, :]  # [B, d_model]
        cls_output = self.output_norm(cls_output)
        
        return cls_output
    
    def get_sequence_output(
        self,
        measurements: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Get full sequence output (not just CLS token).
        
        Useful for interpretability and attention visualization.
        
        Returns:
            Full encoded sequence [batch, seq_len+1, d_model]
        """
        batch_size, seq_len = measurements['cell_ids'].shape
        device = measurements['cell_ids'].device
        
        # Same forward pass but return full sequence
        cell_emb = self.cell_embedding(measurements['cell_ids'])
        beam_emb = self.beam_embedding(measurements['beam_ids'])
        time_emb = self.pos_encoding(measurements['timestamps'])
        
        rt_proj = self.rt_projection(measurements['rt_features'])
        phy_proj = self.phy_projection(measurements['phy_features'])
        mac_proj = self.mac_projection(measurements['mac_features'])
        
        combined = torch.cat([
            cell_emb, beam_emb, time_emb,
            rt_proj, phy_proj, mac_proj,
        ], dim=-1)
        
        tokens = self.input_projection(combined)
        tokens = self.dropout(tokens)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        mask = measurements['mask']
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        src_key_padding_mask = ~extended_mask
        
        encoded = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        encoded = self.output_norm(encoded)
        
        return encoded  # [B, L+1, d_model]
