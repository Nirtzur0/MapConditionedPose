"""
Cross-Attention Fusion Module

Fuses radio encoder output (query) with map encoder spatial tokens (keys/values)
using multi-head cross-attention mechanism.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention between radio measurements and map features.
    
    Radio encoder output acts as query (Q), map spatial tokens as keys/values (K, V).
    This allows the model to attend to relevant map regions based on measurements.
    
    Args:
        d_radio: Dimension of radio encoder output
        d_map: Dimension of map encoder tokens
        d_fusion: Output fusion dimension
        nhead: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_radio: int = 512,
        d_map: int = 768,
        d_fusion: int = 768,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_radio = d_radio
        self.d_map = d_map
        self.d_fusion = d_fusion
        self.nhead = nhead
        
        # Project radio embedding to fusion dimension (for query)
        self.radio_proj = nn.Linear(d_radio, d_fusion)
        
        # Project map tokens if needed (for keys/values)
        self.map_proj = nn.Linear(d_map, d_fusion) if d_map != d_fusion else nn.Identity()
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_fusion,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(d_fusion, d_fusion * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion * 4, d_fusion),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_fusion)
        self.norm2 = nn.LayerNorm(d_fusion)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        radio_emb: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            radio_emb: [batch, d_radio] from radio encoder CLS token
            map_tokens: [batch, num_patches, d_map] from map encoder
        
        Returns:
            Fused representation [batch, d_fusion]
        """
        # Project to fusion dimension
        query = self.radio_proj(radio_emb)  # [B, d_fusion]
        key_value = self.map_proj(map_tokens)  # [B, num_patches, d_fusion]
        
        # Add batch dimension for query (to match attention API)
        query = query.unsqueeze(1)  # [B, 1, d_fusion]
        
        # Cross-attention: radio attends to map
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
        )  # attn_output: [B, 1, d_fusion], attn_weights: [B, 1, num_patches]
        
        # Residual connection + normalization
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        # Remove sequence dimension
        fused = query.squeeze(1)  # [B, d_fusion]
        
        return fused
    
    def forward_with_attention(
        self,
        radio_emb: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning attention weights for visualization.
        
        Returns:
            Fused representation [batch, d_fusion]
            Attention weights [batch, 1, num_patches]
        """
        query = self.radio_proj(radio_emb).unsqueeze(1)
        key_value = self.map_proj(map_tokens)
        
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
        )
        
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        fused = query.squeeze(1)
        
        return fused, attn_weights
