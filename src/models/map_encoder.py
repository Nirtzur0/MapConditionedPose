"""
Map Encoder: Vision Transformer for Radio + OSM Maps

Processes multi-channel maps with early fusion:
- Radio maps: 5 channels (path_gain, toa, snr, sinr, throughput)
- OSM maps: 5 channels (height, material, footprint, road, terrain)

Uses a standard Vision Transformer backbone for map encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from typing import Tuple, Optional, List, Any


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
        in_channels: int = 10,  # Default: 5 radio + 5 OSM
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        radio_map_channels: int = 5,
        osm_map_channels: int = 5,  # All 5 by default
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
