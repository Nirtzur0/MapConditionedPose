"""
Map Encoder: Vision Transformer for Radio + OSM Maps

Processes multi-channel maps with early fusion:
- Radio maps: 5 channels (path_gain, toa, snr, sinr, throughput)
- OSM maps: 5 channels (height, material, footprint, road, terrain)

Uses Vision Transformer (ViT) architecture with patch-based encoding.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them.
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 10,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Convolutional layer for patch extraction and embedding
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [batch, channels, height, width]
        
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
            Grid size (for reshaping later)
        """
        # x: [B, C, H, W]
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]
        
        # Flatten patches
        B, E, H, W = x.shape
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x, H  # Return grid size for potential use


class EquivariantPatchEmbedding(nn.Module):
    """C4-equivariant patch embedding using escnn.
    
    Uses group equivariant convolutions to extract patches while preserving
    rotation equivariance under the C4 group (90-degree rotations).
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension (per rotation)
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 10,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim
        
        # Import escnn here to allow graceful fallback
        try:
            from escnn import gspaces, nn as enn
            self._use_escnn = True
        except ImportError:
            self._use_escnn = False
            
        if self._use_escnn:
            # Define the group space: C4 (90-degree rotations)
            self.gspace = gspaces.rot2dOnR2(N=4)  # C4 group
            
            # Input field type: trivial (scalar) representations for each channel
            self.in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr])
            
            # Output field type: trivial representation 
            # This means each output channel is a scalar field that is INVARIANT to rotation
            # The equivariance comes from the fact that the kernel weights are appropriately constrained
            self.out_type = enn.FieldType(self.gspace, embed_dim * [self.gspace.trivial_repr])
            
            # Equivariant convolution for patch extraction
            self.projection = enn.R2Conv(
                self.in_type,
                self.out_type,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=True,
            )
            
            self._out_channels = embed_dim
        else:
            # Fallback to standard Conv2d
            self.projection = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self._out_channels = embed_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [batch, channels, height, width]
        
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
            Grid size (for reshaping later)
        """
        if self._use_escnn:
            from escnn import nn as enn
            # Wrap input as geometric tensor
            x_geo = enn.GeometricTensor(x, self.in_type)
            
            # Apply equivariant convolution
            out_geo = self.projection(x_geo)
            
            # Extract tensor (features are now C4-equivariant)
            out = out_geo.tensor  # [B, C*4, H/P, W/P]
        else:
            out = self.projection(x)  # [B, embed_dim, H/P, W/P]
        
        # Flatten patches
        B, E, H, W = out.shape
        out = out.flatten(2)  # [B, embed_dim, num_patches]
        out = out.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return out, H


class ViT2DPositionalEncoding(nn.Module):
    """2D sinusoidal positional encoding for spatial tokens."""
    
    def __init__(self, embed_dim: int, grid_size: int):
        super().__init__()
        
        self.grid_size = grid_size
        num_patches = grid_size ** 2
        
        # Create 2D position indices
        y_pos = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size)
        x_pos = torch.arange(grid_size).unsqueeze(0).repeat(grid_size, 1)
        
        # Flatten
        y_pos = y_pos.flatten()  # [num_patches]
        x_pos = x_pos.flatten()  # [num_patches]
        
        # Sinusoidal encoding for x and y separately
        pe = torch.zeros(num_patches, embed_dim)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim // 4, 1) * (-math.log(10000.0) / (embed_dim // 4))
        )
        
        # X position encoding (first half of dimensions)
        pe[:, 0:embed_dim//2:2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 1:embed_dim//2:2] = torch.cos(x_pos.unsqueeze(1) * div_term)
        
        # Y position encoding (second half of dimensions)
        pe[:, embed_dim//2::2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, embed_dim//2+1::2] = torch.cos(y_pos.unsqueeze(1) * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, embed_dim]
        Returns:
            x + positional encoding [batch, num_patches, embed_dim]
        """
        return x + self.pe.unsqueeze(0)





class MapEncoder(nn.Module):
    """Vision Transformer encoder for radio + OSM maps.
    
    Early fusion: concatenate all map channels, then encode with ViT.
    
    Args:
        img_size: Input image size (256x256)
        patch_size: Patch size for ViT (16x16)
        in_channels: Total input channels (5 radio + 5 OSM = 10)
        d_model: Hidden dimension (embed_dim)
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        radio_map_channels: Number of radio map channels (default: 5)
        osm_map_channels: Number of OSM map channels (default: 5)
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 10,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1,
        radio_map_channels: int = 5,
        osm_map_channels: int = 5,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        assert in_channels == radio_map_channels + osm_map_channels, \
            f"in_channels ({in_channels}) must equal radio ({radio_map_channels}) + osm ({osm_map_channels})"
        
        # Patch embedding - use C4-equivariant version
        self.patch_embed = EquivariantPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model,
        )
        
        
        # Positional Encoding: Use simple 2D sinusoidal (compatible with equivariant features)
        # The equivariance is handled by the patch embedding layer itself
        self.pos_encoding = ViT2DPositionalEncoding(d_model, self.grid_size)
        
        # CLS token (optional, for global representation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radio_map: [batch, 5, H, W] (path_gain, toa, snr, sinr, throughput)
            osm_map: [batch, 5, H, W] (height, material, footprint, road, terrain)
        
        Returns:
            Spatial tokens [batch, num_patches, d_model]
            CLS token [batch, d_model] (global representation)
        """
        batch_size = radio_map.shape[0]
        device = radio_map.device
        
        # 1. Early fusion: concatenate radio and OSM maps
        combined_map = torch.cat([radio_map, osm_map], dim=1)  # [B, 10, H, W]
        
        # 2. Patch embedding
        patches, grid_size = self.patch_embed(combined_map)  # [B, num_patches, d_model]
        
        # 3. Add positional encoding
        patches = self.pos_encoding(patches)  # [B, num_patches, d_model]
        patches = self.dropout(patches)
        
        # 4. Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, d_model]
        
        # 5. Transformer encoding
        encoded = self.transformer(tokens)  # [B, num_patches+1, d_model]
        encoded = self.output_norm(encoded)
        
        # 6. Split CLS token and spatial tokens
        cls_output = encoded[:, 0, :]  # [B, d_model]
        spatial_tokens = encoded[:, 1:, :]  # [B, num_patches, d_model]
        
        return spatial_tokens, cls_output
    
    def get_spatial_grid(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> torch.Tensor:
        """Get spatial tokens reshaped as 2D grid.
        
        Useful for visualization and spatial operations.
        
        Returns:
            Spatial grid [batch, d_model, grid_h, grid_w]
        """
        spatial_tokens, _ = self.forward(radio_map, osm_map)
        
        # Reshape to 2D grid
        B, N, D = spatial_tokens.shape
        H = W = int(math.sqrt(N))
        
        grid = spatial_tokens.transpose(1, 2)  # [B, D, N]
        grid = grid.view(B, D, H, W)  # [B, D, H, W]
        
        return grid
    
    def extract_patch(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        center: Tuple[int, int],
        patch_size: int = 64,
    ) -> torch.Tensor:
        """Extract high-resolution patch around a center point.
        
        Used for fine refinement stage.
        
        Args:
            radio_map: [batch, 5, H, W]
            osm_map: [batch, 5, H, W]
            center: (x, y) center coordinates in pixels
            patch_size: Size of patch to extract
        
        Returns:
            Patch embeddings [batch, d_model]
        """
        cx, cy = center
        half_size = patch_size // 2
        
        # Extract patch from both maps
        radio_patch = radio_map[
            :, :,
            max(0, cy - half_size):cy + half_size,
            max(0, cx - half_size):cx + half_size
        ]
        osm_patch = osm_map[
            :, :,
            max(0, cy - half_size):cy + half_size,
            max(0, cx - half_size):cx + half_size
        ]
        
        # Resize if needed (to match expected patch_size)
        if radio_patch.shape[2] != patch_size or radio_patch.shape[3] != patch_size:
            radio_patch = torch.nn.functional.interpolate(
                radio_patch,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False,
            )
            osm_patch = torch.nn.functional.interpolate(
                osm_patch,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False,
            )
        
        # Encode patch (using smaller ViT for efficiency)
        # For now, use same encoder - in practice, might want dedicated fine encoder
        spatial_tokens, cls_token = self.forward(radio_patch, osm_patch)
        
        return cls_token  # [B, d_model]





