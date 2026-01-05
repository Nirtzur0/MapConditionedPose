"""
M3: Map-Conditioned Transformer Models

This package contains the transformer-based UE localization model components.
Includes E2 equivariant vision transformer implementation.
"""

from .radio_encoder import RadioEncoder
from .map_encoder import E2EquivariantMapEncoder, StandardMapEncoder, MapEncoder
from .fusion import CrossAttentionFusion
from .heads import CoarseHead, FineHead
from .ue_localization_model import UELocalizationModel

__all__ = [
    'RadioEncoder',
    'E2EquivariantMapEncoder',
    'StandardMapEncoder',
    'MapEncoder',
    'CrossAttentionFusion',
    'CoarseHead',
    'FineHead',
    'UELocalizationModel',
]
