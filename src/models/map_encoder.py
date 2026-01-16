"""
Map Encoder: E(2) Equivariant Vision Transformer for Radio + OSM Maps

Processes multi-channel maps with early fusion:
- Radio maps: 5 channels (path_gain, toa, snr, sinr, throughput)
- OSM maps: 5 channels (height, material, footprint, road, terrain)

Uses E(2) Equivariant Vision Transformer architecture for rotation/reflection invariance.
Based on GE-ViT: "E(2)-Equivariant Vision Transformer" (Xu et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from typing import Tuple, Optional, List, Any

# E2 equivariant imports
from .e2_vit.g_selfatt.groups import E2
from .e2_vit.g_selfatt.nn import (
    LiftSelfAttention, 
    GroupSelfAttention,
    TransformerBlock,
    LayerNorm,
    Conv3d1x1
)
from .e2_vit.g_selfatt.nn.activations import Swish


def _extract_patch_batch(
    radio_map: torch.Tensor,
    osm_map: torch.Tensor,
    centers_px: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract a batch of patches centered at per-sample pixel coordinates."""
    if centers_px.dim() == 2:
        centers_px = centers_px.unsqueeze(1)

    batch_size, num_patches, _ = centers_px.shape
    height = radio_map.shape[2]
    width = radio_map.shape[3]
    half_low = patch_size // 2
    half_high = patch_size - half_low

    radio_patches = []
    osm_patches = []

    for b in range(batch_size):
        for k in range(num_patches):
            center = centers_px[b, k]
            cx = int(torch.round(center[0]).item())
            cy = int(torch.round(center[1]).item())
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

            x0 = max(0, cx - half_low)
            x1 = min(width, cx + half_high)
            y0 = max(0, cy - half_low)
            y1 = min(height, cy + half_high)

            radio_patch = radio_map[b:b + 1, :, y0:y1, x0:x1]
            osm_patch = osm_map[b:b + 1, :, y0:y1, x0:x1]

            if radio_patch.shape[2] != patch_size or radio_patch.shape[3] != patch_size:
                radio_patch = F.interpolate(
                    radio_patch,
                    size=(patch_size, patch_size),
                    mode='bilinear',
                    align_corners=False,
                )
                osm_patch = F.interpolate(
                    osm_patch,
                    size=(patch_size, patch_size),
                    mode='bilinear',
                    align_corners=False,
                )

            radio_patches.append(radio_patch)
            osm_patches.append(osm_patch)

    if not radio_patches:
        empty_radio = radio_map.new_zeros((0, radio_map.shape[1], patch_size, patch_size))
        empty_osm = osm_map.new_zeros((0, osm_map.shape[1], patch_size, patch_size))
        return empty_radio, empty_osm

    return torch.cat(radio_patches, dim=0), torch.cat(osm_patches, dim=0)


class E2EquivariantMapEncoder(nn.Module):
    """E(2) Equivariant Vision Transformer encoder for radio + OSM maps.
    
    This encoder uses group equivariant self-attention to maintain equivariance
    to rotations and reflections (E(2) group), making the model invariant to
    orientations in the map data.
    
    Based on GE-ViT: "E(2)-Equivariant Vision Transformer" (Xu et al.)
    
    Args:
        img_size: Input image size (256x256)
        in_channels: Total input channels (default: 5 radio + 5 OSM = 10)
        d_model: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of equivariant transformer layers
        num_group_elements: Number of discrete E(2) group elements (default: 8 for p4m)
        dropout: Dropout rate
        radio_map_channels: Number of radio map channels (default: 5)
        osm_map_channels: Number of OSM map channels (default: 5 for backward compat)
        cache_size: Maximum number of cached scenes (0 disables caching)
        cache_mode: When to cache ("off", "eval", "train", "always")
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,  # NEW: patch size to reduce spatial dims
        in_channels: int = 10,  # Default: 5 radio + 5 OSM for backward compat
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        num_group_elements: int = 8,  # p4m group (4 rotations x 2 reflections)
        dropout: float = 0.1,
        radio_map_channels: int = 5,
        osm_map_channels: int = 5,  # All 5 by default for backward compat
        cache_size: int = 0,
        cache_mode: str = "off",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_group_elements = num_group_elements
        self._cache_size = cache_size
        self._cache_mode = cache_mode if cache_mode in {"off", "eval", "train", "always"} else "off"
        self._cache = OrderedDict()
        
        # Calculate patch grid dimensions
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        assert in_channels == radio_map_channels + osm_map_channels, \
            f"in_channels ({in_channels}) must equal radio ({radio_map_channels}) + osm ({osm_map_channels})"
        
        # Patch embedding - reduces spatial dimensions BEFORE E2 attention
        self.patch_embed = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Initialize E(2) group
        self.group = E2(num_elements=num_group_elements)
        
        # Lifting layer: lifts from spatial domain to group domain
        # CRITICAL FIX: Use patch-level spatial dimensions (num_patches_per_side)
        # instead of pixel-level (img_size) to avoid 256GB memory requirement!
        # For 256x256 image with 16x16 patches: 16 spatial dims vs 256 spatial dims
        mid_channels = d_model // 2
        self.lifting_layer = LiftSelfAttention(
            group=self.group,
            in_channels=d_model,  # After patch embedding
            mid_channels=mid_channels,
            out_channels=d_model,
            num_heads=num_heads,
            max_pos_embedding=self.num_patches_per_side,  # FIXED: 16 instead of 256!
            attention_dropout_rate=dropout,
        )
        
        # Normalization and activation after lifting
        self.lifting_norm = LayerNorm(d_model)
        self.lifting_activation = Swish()
        
        # Group equivariant transformer layers
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            attention_layer = GroupSelfAttention(
                group=self.group,
                in_channels=d_model,
                mid_channels=d_model // 2,
                out_channels=d_model,
                num_heads=num_heads,
                max_pos_embedding=self.num_patches_per_side,  # FIXED: Use patch dims!
                attention_dropout_rate=dropout,
            )
            
            transformer_block = TransformerBlock(
                in_channels=d_model,
                out_channels=d_model,
                attention_layer=attention_layer,
                norm_type="LayerNorm",
                activation_function="Swish",
                crop_size=0,  # No cropping for full image
                value_dropout_rate=dropout,
                dim_mlp_conv=3,  # 3D convolutions for group signals
            )
            self.transformer_blocks.append(transformer_block)
        
        # Final normalization
        self.output_norm = LayerNorm(d_model)
        
        # Pooling to get fixed-size output
        # Average pool over spatial dimensions
        self.spatial_pool = nn.AdaptiveAvgPool3d((num_group_elements, 1, 1))
        
        # Group pooling: sum over group dimension for invariance
        # Or we can use max pooling for more robustness
        self.group_pool_type = 'max'  # 'sum' or 'max'
        
        # Optional: Project to fixed dimension after pooling
        self.output_projection = nn.Linear(d_model * num_group_elements if self.group_pool_type == 'concat' else d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                # Whitening initialization as in original GE-ViT
                m.weight.data.normal_(
                    0,
                    1.41421356 * torch.prod(torch.Tensor(list(m.weight.shape)[1:])) ** (-1 / 2),
                )
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _cache_active(self, cache_keys: Optional[List[Any]]) -> bool:
        if self._cache_size <= 0 or self._cache_mode == "off" or cache_keys is None:
            return False
        if self.training:
            if self._cache_mode not in {"train", "always"}:
                return False
            if any(param.requires_grad for param in self.parameters()):
                return False
        else:
            if self._cache_mode not in {"eval", "always"}:
                return False
        return True

    @staticmethod
    def _normalize_cache_keys(
        cache_keys: Optional[Any],
        batch_size: int,
    ) -> Optional[List[Any]]:
        if cache_keys is None:
            return None
        if isinstance(cache_keys, torch.Tensor):
            if cache_keys.numel() != batch_size:
                return None
            keys = cache_keys.detach().view(-1).tolist()
        elif isinstance(cache_keys, (list, tuple)):
            if len(cache_keys) != batch_size:
                return None
            keys = list(cache_keys)
        else:
            return None

        normalized: List[Any] = []
        for key in keys:
            if key is None:
                normalized.append(None)
                continue
            if isinstance(key, torch.Tensor):
                key = key.item()
            if isinstance(key, bool):
                normalized.append(int(key))
            elif isinstance(key, int):
                normalized.append(key if key >= 0 else None)
            elif isinstance(key, str):
                normalized.append(key if key else None)
            else:
                try:
                    key_int = int(key)
                except (TypeError, ValueError):
                    normalized.append(None)
                else:
                    normalized.append(key_int if key_int >= 0 else None)
        return normalized

    def _get_cached(self, key: Any, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cached = self._cache.get(key)
        if cached is None:
            return None
        tokens, cls_token = cached
        if tokens.device != device:
            tokens = tokens.to(device)
            cls_token = cls_token.to(device)
            self._cache[key] = (tokens, cls_token)
        self._cache.move_to_end(key)
        return tokens, cls_token

    def _set_cache(self, key: Any, tokens: torch.Tensor, cls_token: torch.Tensor) -> None:
        if key is None or self._cache_size <= 0:
            return
        self._cache[key] = (tokens.detach(), cls_token.detach())
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _encode_maps(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Early fusion: concatenate radio and OSM maps
        combined_map = torch.cat([radio_map, osm_map], dim=1)  # [B, 10, H, W]

        # 2. Patch embedding - reduces spatial dimensions
        # [B, 10, H, W] -> [B, d_model, H/P, W/P] where P=patch_size
        patched = self.patch_embed(combined_map)

        # 3. Lift to group domain (now operating on reduced spatial dims!)
        # Output: [B, d_model, num_group_elements, H/P, W/P]
        lifted = self.lifting_layer(patched)
        lifted = self.lifting_norm(lifted)
        lifted = self.lifting_activation(lifted)

        # 4. Apply equivariant transformer blocks
        out = lifted
        for block in self.transformer_blocks:
            out = block(out)

        # 4. Final normalization
        out = self.output_norm(out)  # [B, d_model, G, H, W]

        # 5. Pool to get fixed-size representations
        # First, pool spatially
        B, C, G, H, W = out.shape
        out_spatial_pooled = self.spatial_pool(out)  # [B, d_model, G, 1, 1]
        out_spatial_pooled = out_spatial_pooled.squeeze(-1).squeeze(-1)  # [B, d_model, G]

        # For cls_token: pool over group dimension for invariance
        if self.group_pool_type == 'sum':
            cls_token = out_spatial_pooled.sum(dim=2)  # [B, d_model]
        elif self.group_pool_type == 'max':
            cls_token = out_spatial_pooled.max(dim=2)[0]  # [B, d_model]
        elif self.group_pool_type == 'concat':
            cls_token = out_spatial_pooled.flatten(1)  # [B, d_model * G]
        else:
            cls_token = out_spatial_pooled.mean(dim=2)  # [B, d_model]

        cls_token = self.output_projection(cls_token)  # [B, d_model]

        # For spatial_tokens: flatten spatial and group dimensions
        # This gives position-dependent features while maintaining equivariance
        out_flattened = out.permute(0, 3, 4, 2, 1)  # [B, H, W, G, C]
        out_flattened = out_flattened.reshape(B, H * W, G * C)  # [B, H*W, G*C]

        # Project back to d_model dimension if needed
        if G * C != self.d_model:
            # Use a learnable projection
            if not hasattr(self, 'spatial_projection'):
                self.spatial_projection = nn.Linear(G * C, self.d_model).to(out.device)
            spatial_tokens = self.spatial_projection(out_flattened)
        else:
            spatial_tokens = out_flattened

        return spatial_tokens, cls_token

    def forward(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        cache_keys: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radio_map: [batch, 5, H, W] (path_gain, toa, snr, sinr, throughput)
            osm_map: [batch, 5, H, W] (height, material, footprint, road, terrain)
            cache_keys: Optional scene keys for caching map encodings

        Returns:
            spatial_tokens: [batch, num_patches, d_model] - patch-level features
            cls_token: [batch, d_model] - global representation (pooled)
        """
        batch_size = radio_map.shape[0]
        normalized_keys = self._normalize_cache_keys(cache_keys, batch_size)

        if not self._cache_active(normalized_keys):
            return self._encode_maps(radio_map, osm_map)

        tokens_out: List[Optional[torch.Tensor]] = [None] * batch_size
        cls_out: List[Optional[torch.Tensor]] = [None] * batch_size
        compute_indices: List[int] = []

        for i, key in enumerate(normalized_keys):
            if key is None:
                compute_indices.append(i)
                continue
            cached = self._get_cached(key, radio_map.device)
            if cached is None:
                compute_indices.append(i)
            else:
                tokens_out[i], cls_out[i] = cached

        if compute_indices:
            radio_batch = radio_map[compute_indices]
            osm_batch = osm_map[compute_indices]
            with torch.no_grad():
                tokens_batch, cls_batch = self._encode_maps(radio_batch, osm_batch)
            for j, idx in enumerate(compute_indices):
                tokens_out[idx] = tokens_batch[j]
                cls_out[idx] = cls_batch[j]
                key = normalized_keys[idx]
                if key is not None:
                    self._set_cache(key, tokens_out[idx], cls_out[idx])

        return torch.stack(tokens_out, dim=0), torch.stack(cls_out, dim=0)
    
    def get_spatial_grid(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> torch.Tensor:
        """Get spatial tokens reshaped as 2D grid.
        
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
        
        # Resize if needed
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
        
        # Encode patch
        _, cls_token = self.forward(radio_patch, osm_patch)
        
        return cls_token  # [B, d_model]

    def encode_local_patches(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        centers_px: torch.Tensor,
        patch_size: int = 64,
    ) -> torch.Tensor:
        """Encode local patches centered at per-sample pixel coordinates.

        Args:
            radio_map: [B, C, H, W]
            osm_map: [B, C, H, W]
            centers_px: [B, K, 2] or [B, 2] center coords in pixels (x, y)
            patch_size: Patch size in pixels

        Returns:
            patch_embeddings: [B, K, d_model]
        """
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if centers_px.dim() == 2:
            centers_px = centers_px.unsqueeze(1)

        batch_size, num_patches, _ = centers_px.shape
        radio_patches, osm_patches = _extract_patch_batch(
            radio_map,
            osm_map,
            centers_px,
            patch_size,
        )

        if radio_patches.shape[0] == 0:
            return radio_map.new_zeros((batch_size, num_patches, self.d_model))

        _, cls_tokens = self._encode_maps(radio_patches, osm_patches)
        return cls_tokens.view(batch_size, num_patches, -1)


class StandardMapEncoder(nn.Module):
    """Standard Vision Transformer encoder for radio + OSM maps (non-equivariant).
    
    Uses 10 input channels by default: 5 radio (path_gain, toa, snr, sinr, throughput)
    + 5 OSM (height, material, footprint, road, terrain).
    
    Note: Material (1), Road (3), and Terrain (4) OSM channels are often constant/empty
    in Sionna-generated scenes. For new training, consider using osm_channels=[0,2]
    in the dataset to filter to only Height and Footprint.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 10,  # Default: 5 radio + 5 OSM for backward compat
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        radio_map_channels: int = 5,
        osm_map_channels: int = 5,  # All 5 by default for backward compat
        cache_size: int = 0,
        cache_mode: str = "off",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        self._cache_size = cache_size
        self._cache_mode = cache_mode if cache_mode in {"off", "eval", "train", "always"} else "off"
        self._cache = OrderedDict()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, 
            d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model) * 0.02)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def _interpolate_pos_embed(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Interpolate positional embeddings to match a new patch grid size."""
        pos_cls = self.pos_embed[:, :1]
        pos_patch = self.pos_embed[:, 1:]
        base_size = int(self.num_patches ** 0.5)

        pos_patch = pos_patch.reshape(1, base_size, base_size, self.d_model).permute(0, 3, 1, 2)
        pos_patch = F.interpolate(
            pos_patch,
            size=(grid_h, grid_w),
            mode='bicubic',
            align_corners=False,
        )
        pos_patch = pos_patch.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.d_model)
        return torch.cat([pos_cls, pos_patch], dim=1)
        
    def _cache_active(self, cache_keys: Optional[List[Any]]) -> bool:
        if self._cache_size <= 0 or self._cache_mode == "off" or cache_keys is None:
            return False
        if self.training:
            if self._cache_mode not in {"train", "always"}:
                return False
            if any(param.requires_grad for param in self.parameters()):
                return False
        else:
            if self._cache_mode not in {"eval", "always"}:
                return False
        return True

    @staticmethod
    def _normalize_cache_keys(
        cache_keys: Optional[Any],
        batch_size: int,
    ) -> Optional[List[Any]]:
        if cache_keys is None:
            return None
        if isinstance(cache_keys, torch.Tensor):
            if cache_keys.numel() != batch_size:
                return None
            keys = cache_keys.detach().view(-1).tolist()
        elif isinstance(cache_keys, (list, tuple)):
            if len(cache_keys) != batch_size:
                return None
            keys = list(cache_keys)
        else:
            return None

        normalized: List[Any] = []
        for key in keys:
            if key is None:
                normalized.append(None)
                continue
            if isinstance(key, torch.Tensor):
                key = key.item()
            if isinstance(key, bool):
                normalized.append(int(key))
            elif isinstance(key, int):
                normalized.append(key if key >= 0 else None)
            elif isinstance(key, str):
                normalized.append(key if key else None)
            else:
                try:
                    key_int = int(key)
                except (TypeError, ValueError):
                    normalized.append(None)
                else:
                    normalized.append(key_int if key_int >= 0 else None)
        return normalized

    def _get_cached(self, key: Any, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cached = self._cache.get(key)
        if cached is None:
            return None
        tokens, cls_token = cached
        if tokens.device != device:
            tokens = tokens.to(device)
            cls_token = cls_token.to(device)
            self._cache[key] = (tokens, cls_token)
        self._cache.move_to_end(key)
        return tokens, cls_token

    def _set_cache(self, key: Any, tokens: torch.Tensor, cls_token: torch.Tensor) -> None:
        if key is None or self._cache_size <= 0:
            return
        self._cache[key] = (tokens.detach(), cls_token.detach())
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _encode_maps(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = radio_map.shape[0]

        # Concatenate maps
        combined_map = torch.cat([radio_map, osm_map], dim=1)  # [B, 10, H, W]

        # Patch embedding
        x = self.patch_embed(combined_map)  # [B, d_model, H/P, W/P]
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, d_model]

        # Add positional embedding
        pos_embed = self.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = self._interpolate_pos_embed(grid_h, grid_w)
        pos_embed = pos_embed.to(device=x.device, dtype=x.dtype)
        x = x + pos_embed

        # Transformer
        x = self.transformer(x)  # [B, N+1, d_model]

        # Layer norm
        x = self.norm(x)

        # Split class token and patch tokens
        cls_token = x[:, 0]  # [B, d_model]
        tokens = x[:, 1:]    # [B, N, d_model]

        return tokens, cls_token

    def forward(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        cache_keys: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            radio_map: [B, 5, H, W]
            osm_map: [B, 5, H, W]
            cache_keys: Optional scene keys for caching map encodings

        Returns:
            tokens: [B, N, d_model] - all patch tokens
            cls_token: [B, d_model] - classification token
        """
        batch_size = radio_map.shape[0]
        normalized_keys = self._normalize_cache_keys(cache_keys, batch_size)

        if not self._cache_active(normalized_keys):
            return self._encode_maps(radio_map, osm_map)

        tokens_out: List[Optional[torch.Tensor]] = [None] * batch_size
        cls_out: List[Optional[torch.Tensor]] = [None] * batch_size
        compute_indices: List[int] = []

        for i, key in enumerate(normalized_keys):
            if key is None:
                compute_indices.append(i)
                continue
            cached = self._get_cached(key, radio_map.device)
            if cached is None:
                compute_indices.append(i)
            else:
                tokens_out[i], cls_out[i] = cached

        if compute_indices:
            radio_batch = radio_map[compute_indices]
            osm_batch = osm_map[compute_indices]
            with torch.no_grad():
                tokens_batch, cls_batch = self._encode_maps(radio_batch, osm_batch)
            for j, idx in enumerate(compute_indices):
                tokens_out[idx] = tokens_batch[j]
                cls_out[idx] = cls_batch[j]
                key = normalized_keys[idx]
                if key is not None:
                    self._set_cache(key, tokens_out[idx], cls_out[idx])

        return torch.stack(tokens_out, dim=0), torch.stack(cls_out, dim=0)

    def extract_patch(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        center: Tuple[int, int],
        patch_size: int = 64,
    ) -> torch.Tensor:
        """Extract and encode a patch around a shared center for the batch."""
        center_tensor = radio_map.new_tensor(center, dtype=torch.float32)
        center_tensor = center_tensor.unsqueeze(0).expand(radio_map.shape[0], -1)
        patch_embeddings = self.encode_local_patches(
            radio_map,
            osm_map,
            centers_px=center_tensor,
            patch_size=patch_size,
        )
        return patch_embeddings[:, 0, :]

    def encode_local_patches(
        self,
        radio_map: torch.Tensor,
        osm_map: torch.Tensor,
        centers_px: torch.Tensor,
        patch_size: int = 64,
    ) -> torch.Tensor:
        """Encode local patches centered at per-sample pixel coordinates."""
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if centers_px.dim() == 2:
            centers_px = centers_px.unsqueeze(1)

        batch_size, num_patches, _ = centers_px.shape
        radio_patches, osm_patches = _extract_patch_batch(
            radio_map,
            osm_map,
            centers_px,
            patch_size,
        )

        if radio_patches.shape[0] == 0:
            return radio_map.new_zeros((batch_size, num_patches, self.d_model))

        _, cls_tokens = self._encode_maps(radio_patches, osm_patches)
        return cls_tokens.view(batch_size, num_patches, -1)


# Alias for backward compatibility
MapEncoder = StandardMapEncoder
