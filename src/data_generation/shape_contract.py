"""
Canonical shape contracts and normalization utilities for data generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ShapeContractError(ValueError):
    """Raised when inputs cannot be normalized to canonical shapes."""


def _is_tensor(x: Any) -> bool:
    return TF_AVAILABLE and isinstance(x, (tf.Tensor, tf.Variable))


@dataclass(frozen=True)
class ChannelMatrix:
    """
    Canonical channel matrix shape: [B, Rx, RxAnt, C, TxAnt, F]
    """
    data: Any
    batch: int
    rx: int
    rx_ant: int
    cells: int
    tx_ant: int
    freq: int


def _to_numpy_shape(x: Any) -> tuple:
    if _is_tensor(x):
        shape = x.shape
        return tuple(int(d) if d is not None else -1 for d in shape)
    return tuple(x.shape)


def normalize_channel_matrix(
    channel_matrix: Any,
    *,
    strict: bool = False,
) -> ChannelMatrix:
    """
    Normalize channel matrix to canonical shape [B, Rx, RxAnt, C, TxAnt, F].

    Accepted input shapes:
      - [B, Rx, RxAnt, C, TxAnt, F] (already canonical)
      - [B, RxAnt, C, TxAnt, F] (Rx dimension missing)
    """
    if channel_matrix is None:
        raise ShapeContractError("channel_matrix is None")

    if _is_tensor(channel_matrix):
        rank = len(channel_matrix.shape)
        if rank == 6:
            data = channel_matrix
        elif rank == 5:
            data = tf.expand_dims(channel_matrix, axis=1)
        else:
            msg = f"Unsupported channel_matrix rank {rank} (expected 5 or 6)"
            if strict:
                raise ShapeContractError(msg)
            return ChannelMatrix(channel_matrix, -1, -1, -1, -1, -1, -1)
        shape = _to_numpy_shape(data)
    else:
        rank = channel_matrix.ndim
        if rank == 6:
            data = channel_matrix
        elif rank == 5:
            data = np.expand_dims(channel_matrix, axis=1)
        else:
            msg = f"Unsupported channel_matrix rank {rank} (expected 5 or 6)"
            if strict:
                raise ShapeContractError(msg)
            return ChannelMatrix(channel_matrix, -1, -1, -1, -1, -1, -1)
        shape = _to_numpy_shape(data)

    batch, rx, rx_ant, cells, tx_ant, freq = shape
    return ChannelMatrix(
        data=data,
        batch=batch,
        rx=rx,
        rx_ant=rx_ant,
        cells=cells,
        tx_ant=tx_ant,
        freq=freq,
    )


def normalize_sinr(sinr: Any) -> Any:
    """
    Normalize SINR to shape [B, Rx, C]. Accepts [B, C] and [B, Rx, C].
    """
    if sinr is None:
        return None
    if _is_tensor(sinr):
        rank = len(sinr.shape)
        if rank == 2:
            return tf.expand_dims(sinr, axis=1)
        if rank == 3:
            return sinr
        return sinr
    rank = sinr.ndim
    if rank == 2:
        return np.expand_dims(sinr, axis=1)
    return sinr


def normalize_rsrp(rsrp: Any) -> Any:
    """
    Normalize RSRP to shape [B, Rx, C]. Accepts [B, C] and [B, Rx, C].
    """
    return normalize_sinr(rsrp)
