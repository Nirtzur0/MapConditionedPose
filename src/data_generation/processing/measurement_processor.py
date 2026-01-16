"""
Measurement realism processing (dropout, quantization).
"""

import logging
import numpy as np
from typing import Dict

logger = logging.getLogger(__name__)


class MeasurementProcessor:
    """
    Applies measurement realism to simulated data.
    
    Responsibilities:
    - Apply measurement dropout
    - Apply quantization
    - Simulate realistic measurement errors
    """
    
    def __init__(self, config: any):
        """
        Initialize measurement processor.
        
        Args:
            config: Data generation configuration
        """
        self.config = config
        seed = getattr(config, 'measurement_dropout_seed', 42)
        self._dropout_rng = np.random.default_rng(seed)
        
    def apply_realism(self, scene_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply measurement dropout and quantization.
        
        Args:
            scene_data: Dictionary of scene features
            
        Returns:
            Updated scene data with realism applied
        """
        if not self.config.quantization_enabled:
            return scene_data
        
        # Apply dropout to PHY/FAPI measurements
        phy_fapi_keys = [k for k in scene_data.keys() if k.startswith('phy_fapi/')]
        phy_fapi_dict = {k: scene_data[k] for k in phy_fapi_keys if isinstance(scene_data[k], np.ndarray)}
        
        # Map keys to dropout rates
        dropout_mapping = {
            'phy_fapi/rsrp': 'rsrp',
            'phy_fapi/rsrq': 'rsrq',
            'phy_fapi/sinr': 'sinr',
            'phy_fapi/cqi': 'cqi',
            'phy_fapi/ri': 'ri',
            'phy_fapi/pmi': 'pmi',
        }
        
        dropout_rates = {
            k: self.config.measurement_dropout_rates.get(v, 0.0) if self.config.measurement_dropout_rates else 0.0
            for k, v in dropout_mapping.items()
        }
        
        # Apply dropout
        phy_fapi_dropped = self._add_measurement_dropout(phy_fapi_dict, dropout_rates)
        
        # Update scene_data
        scene_data.update(phy_fapi_dropped)
        
        return scene_data
    
    def _add_measurement_dropout(
        self,
        data_dict: Dict[str, np.ndarray],
        dropout_rates: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Add measurement dropout to data.
        
        Args:
            data_dict: Dictionary of measurement arrays
            dropout_rates: Dropout rate for each measurement type
            
        Returns:
            Dictionary with dropout applied
        """
        # Import the actual dropout function
        try:
            from ..measurement_utils import add_measurement_dropout
            return add_measurement_dropout(data_dict, dropout_rates, rng=self._dropout_rng)
        except ImportError:
            logger.warning("measurement_utils not available, skipping dropout")
            return data_dict
