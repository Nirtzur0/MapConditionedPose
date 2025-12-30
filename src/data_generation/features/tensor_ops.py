"""
Tensor Operations Abstraction
Abstract base class and implementations for tensor operations (NumPy/TensorFlow compatibility).
"""

import numpy as np
from typing import List, Tuple, Any
import logging

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)

class TensorOps:
    """Abstract base class for tensor operations (NumPy/TensorFlow compatibility)."""
    
    def is_tensor(self, x: Any) -> bool:
        raise NotImplementedError
    
    def to_numpy(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    def abs(self, x: Any) -> Any:
        raise NotImplementedError
        
    def log10(self, x: Any) -> Any:
        raise NotImplementedError
        
    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        raise NotImplementedError

    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        raise NotImplementedError
        
    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        raise NotImplementedError
        
    def sqrt(self, x: Any) -> Any:
        raise NotImplementedError

    def square(self, x: Any) -> Any:
        raise NotImplementedError
        
    def expand_dims(self, x: Any, axis: int) -> Any:
        raise NotImplementedError
        
    def power(self, x: Any, y: Any) -> Any:
        raise NotImplementedError

    def transpose(self, x: Any, perm: List[int]) -> Any:
        raise NotImplementedError

    def complex(self, real: Any, imag: Any) -> Any:
        raise NotImplementedError
        
    def shape(self, x: Any) -> Tuple[int, ...]:
        raise NotImplementedError
        
    def clip(self, x: Any, min_val: float, max_val: float) -> Any:
        raise NotImplementedError
        
    def pad(self, x: Any, paddings: List[List[int]], mode: str, constant_values: Any) -> Any:
        raise NotImplementedError

class NumpyOps(TensorOps):
    def is_tensor(self, x: Any) -> bool: return False
    def to_numpy(self, x: Any) -> np.ndarray:
        if hasattr(x, 'numpy'):
            return x.numpy()
        return np.array(x) if not isinstance(x, np.ndarray) else x
    def abs(self, x: Any) -> Any: return np.abs(x)
    def log10(self, x: Any) -> Any: return np.log10(x)
    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return np.mean(x, axis=axis, keepdims=keepdims)
    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return np.sum(x, axis=axis, keepdims=keepdims)
    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return np.max(x, axis=axis, keepdims=keepdims)
    def sqrt(self, x: Any) -> Any: return np.sqrt(x)
    def square(self, x: Any) -> Any: return np.square(x)
    def expand_dims(self, x: Any, axis: int) -> Any: return np.expand_dims(x, axis)
    def power(self, x: Any, y: Any) -> Any: return np.power(x, y)
    def transpose(self, x: Any, perm: List[int]) -> Any: return np.transpose(x, axes=perm)
    def complex(self, real: Any, imag: Any) -> Any: return self.to_numpy(real) + 1j * self.to_numpy(imag)
    def shape(self, x: Any) -> Tuple[int, ...]: return x.shape if hasattr(x, 'shape') else (len(x),)
    def clip(self, x: Any, min_val: float, max_val: float) -> Any: return np.clip(x, min_val, max_val)
    def pad(self, x: Any, paddings: List[List[int]], mode: str, constant_values: Any) -> Any:
        # np.pad expects tuple of tuples
        pad_width = [(p[0], p[1]) for p in paddings]
        return np.pad(x, pad_width, mode=mode.lower(), constant_values=constant_values)


class TFOps(TensorOps):
    def is_tensor(self, x: Any) -> bool: return tf.is_tensor(x)
    def to_numpy(self, x: Any) -> np.ndarray: return x.numpy()
    def abs(self, x: Any) -> Any: return tf.abs(x)
    def log10(self, x: Any) -> Any: return tf.math.log(x) / tf.math.log(10.0)
    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: return tf.reduce_max(x, axis=axis, keepdims=keepdims)
    def sqrt(self, x: Any) -> Any: return tf.sqrt(x)
    def square(self, x: Any) -> Any: return tf.square(x)
    def expand_dims(self, x: Any, axis: int) -> Any: return tf.expand_dims(x, axis)
    def power(self, x: Any, y: Any) -> Any: return tf.pow(x, y)
    def transpose(self, x: Any, perm: List[int]) -> Any: return tf.transpose(x, perm=perm)
    def complex(self, real: Any, imag: Any) -> Any: return tf.complex(real, imag)
    def shape(self, x: Any) -> Tuple[int, ...]: return tuple(x.shape)
    def clip(self, x: Any, min_val: float, max_val: float) -> Any: return tf.clip_by_value(x, min_val, max_val)
    def pad(self, x: Any, paddings: List[List[int]], mode: str, constant_values: Any) -> Any:
        return tf.pad(x, paddings, mode=mode, constant_values=constant_values)

def get_ops(data: Any) -> TensorOps:
    """Factory to get the correct operations backend."""
    if TF_AVAILABLE and (tf.is_tensor(data) or (hasattr(data, 'a') and tf.is_tensor(data.a))):
        return TFOps()
    return NumpyOps()

def _to_numpy(v: Any) -> np.ndarray:
    """Helper to convert Any (Tensor or Array) to NumPy array."""
    if v is None:
        return None
    if hasattr(v, 'numpy'):
        return v.numpy()
    if isinstance(v, (list, tuple)):
        return np.array(v)
    return v
