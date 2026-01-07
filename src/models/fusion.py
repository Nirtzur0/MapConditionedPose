"""
Cross-Attention Fusion Module

Fuses radio encoder output (query) with map encoder spatial tokens (keys/values)
using multi-head cross-attention mechanism.

Improvements:
- Uses multiple learnable query tokens for richer fusion
- Two-stage attention: radio->map, then aggregate
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-attention between radio measurements and map features.
    
    Radio encoder output acts as query (Q), map spatial tokens as keys/values (K, V).
    This allows the model to attend to relevant map regions based on measurements.
    
    Uses multiple learnable query tokens for richer fusion capacity.
    
    Args:
        d_radio: Dimension of radio encoder output
        d_map: Dimension of map encoder tokens
        d_fusion: Output fusion dimension
        nhead: Number of attention heads
        dropout: Dropout rate
        num_query_tokens: Number of learnable query tokens (default: 4)
    """
    
    def __init__(
        self,
        d_radio: int = 512,
        d_map: int = 768,
        d_fusion: int = 768,
        nhead: int = 8,
        dropout: float = 0.1,
        num_query_tokens: int = 4,
    ):
        super().__init__()
        
        self.d_radio = d_radio
        self.d_map = d_map
        self.d_fusion = d_fusion
        self.nhead = nhead
        self.num_query_tokens = num_query_tokens
        
        # Project radio embedding to fusion dimension (for query)
        self.radio_proj = nn.Linear(d_radio, d_fusion)
        
        # Learnable query tokens for multi-query attention
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_fusion) * 0.02)
        
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
        self.norm_out = nn.LayerNorm(d_fusion)
        
        # Aggregation: combine radio token with multi-query outputs
        self.aggregate = nn.Linear(d_fusion * (num_query_tokens + 1), d_fusion)
        
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
        B = radio_emb.shape[0]
        
        # Project radio to fusion dimension
        radio_proj = self.radio_proj(radio_emb)  # [B, d_fusion]
        key_value = self.map_proj(map_tokens)  # [B, num_patches, d_fusion]
        
        # Expand learnable query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_query_tokens, d_fusion]
        
        # Condition queries on radio embedding (additive conditioning)
        queries = queries + radio_proj.unsqueeze(1)  # [B, num_query_tokens, d_fusion]
        
        # Cross-attention: queries attend to map
        attn_output, _ = self.cross_attention(
            query=queries,
            key=key_value,
            value=key_value,
            need_weights=False,
        )  # attn_output: [B, num_query_tokens, d_fusion]
        
        # Residual connection + normalization
        queries = self.norm1(queries + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(queries)
        queries = self.norm2(queries + ffn_output)  # [B, num_query_tokens, d_fusion]
        
        # Aggregate all query tokens with radio embedding
        # Flatten query tokens and concatenate with radio projection
        queries_flat = queries.flatten(1)  # [B, num_query_tokens * d_fusion]
        combined = torch.cat([radio_proj, queries_flat], dim=1)  # [B, (1 + num_query_tokens) * d_fusion]
        
        fused = self.aggregate(combined)  # [B, d_fusion]
        fused = self.norm_out(fused)
        
        return fused
    
    def forward_with_attention(
        self,
        radio_emb: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning attention weights for visualization.
        
        Returns:
            Fused representation [batch, d_fusion]
            Attention weights [batch, num_query_tokens, num_patches]
        """
        B = radio_emb.shape[0]
        
        radio_proj = self.radio_proj(radio_emb)
        key_value = self.map_proj(map_tokens)
        
        queries = self.query_tokens.expand(B, -1, -1)
        queries = queries + radio_proj.unsqueeze(1)
        
        attn_output, attn_weights = self.cross_attention(
            query=queries,
            key=key_value,
            value=key_value,
            need_weights=True,
        )
        
        queries = self.norm1(queries + self.dropout(attn_output))
        ffn_output = self.ffn(queries)
        queries = self.norm2(queries + ffn_output)
        
        queries_flat = queries.flatten(1)
        combined = torch.cat([radio_proj, queries_flat], dim=1)
        
        fused = self.aggregate(combined)
        fused = self.norm_out(fused)
        
        # Average attention weights across query tokens for visualization
        attn_weights_avg = attn_weights.mean(dim=1, keepdim=True)  # [B, 1, num_patches]
        
        return fused, attn_weights_avg
