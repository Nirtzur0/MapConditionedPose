"""
Material Domain Randomization
Handles material property sampling for robust sim-to-real transfer
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MaterialRandomizer:
    """
    Material domain randomization for robust training.
    
    Samples material properties with physical constraints based on ITU-R P.2040
    and 3GPP channel models.
    """
    
    # Material ranges based on ITU-R P.2040-1 (f-dependent)
    MATERIAL_CONFIGS = {
        # Ground materials
        'wet_ground': {
            'itu_id': 'mat-itu_wet_ground',
            'epsilon_r': (10.0, 30.0),  # Relative permittivity
            'sigma': (0.01, 0.1),       # Conductivity [S/m]
            'freq_range_ghz': (1.0, 10.0),
            'category': 'ground',
        },
        'medium_dry_ground': {
            'itu_id': 'mat-itu_medium_dry_ground',
            'epsilon_r': (4.0, 8.0),
            'sigma': (0.001, 0.01),
            'freq_range_ghz': (1.0, 10.0),
            'category': 'ground',
        },
        'very_dry_ground': {
            'itu_id': 'mat-itu_very_dry_ground',
            'epsilon_r': (2.5, 4.0),
            'sigma': (0.0001, 0.001),
            'freq_range_ghz': (1.0, 10.0),
            'category': 'ground',
        },
        
        # Building materials
        'concrete': {
            'itu_id': 'mat-itu_concrete',
            'epsilon_r': (5.0, 8.0),
            'sigma': (0.01, 0.1),
            'freq_range_ghz': (1.0, 100.0),
            'category': 'building',
        },
        'brick': {
            'itu_id': 'mat-itu_brick',
            'epsilon_r': (3.5, 5.5),
            'sigma': (0.005, 0.05),
            'freq_range_ghz': (1.0, 100.0),
            'category': 'building',
        },
        'wood': {
            'itu_id': 'mat-itu_wood',
            'epsilon_r': (1.5, 3.0),
            'sigma': (0.001, 0.01),
            'freq_range_ghz': (1.0, 100.0),
            'category': 'building',
        },
        'glass': {
            'itu_id': 'mat-itu_glass',
            'epsilon_r': (6.0, 8.0),
            'sigma': (0.0, 0.001),
            'freq_range_ghz': (1.0, 100.0),
            'category': 'building',
        },
        
        # Rooftop materials
        'metal': {
            'itu_id': 'mat-itu_metal',
            'epsilon_r': (1.0, 1.0),  # Perfect conductor approximation
            'sigma': (1e6, 1e7),
            'freq_range_ghz': (1.0, 100.0),
            'category': 'rooftop',
        },
    }
    
    def __init__(
        self,
        enable_randomization: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize material randomizer.
        
        Args:
            enable_randomization: If False, always return default materials
            seed: Random seed for reproducibility
        """
        self.enable_randomization = enable_randomization
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        logger.info(f"MaterialRandomizer initialized (randomization={'ON' if enable_randomization else 'OFF'})")
    
    def sample(self) -> Dict[str, str]:
        """
        Sample material combination for a scene.
        
        Returns:
            Dict with 'ground', 'rooftop', 'wall' ITU material IDs
        """
        if not self.enable_randomization:
            return {
                'ground': 'mat-itu_very_dry_ground',
                'rooftop': 'mat-itu_concrete',
                'wall': 'mat-itu_brick',
            }
        
        # Sample from appropriate categories
        ground_materials = [m for m, cfg in self.MATERIAL_CONFIGS.items() 
                          if cfg['category'] == 'ground']
        building_materials = [m for m, cfg in self.MATERIAL_CONFIGS.items() 
                            if cfg['category'] == 'building']
        rooftop_materials = [m for m, cfg in self.MATERIAL_CONFIGS.items() 
                           if cfg['category'] == 'rooftop']
        
        ground = self.rng.choice(ground_materials)
        wall = self.rng.choice(building_materials)
        rooftop = self.rng.choice(rooftop_materials)
        
        materials = {
            'ground': self.MATERIAL_CONFIGS[ground]['itu_id'],
            'rooftop': self.MATERIAL_CONFIGS[rooftop]['itu_id'],
            'wall': self.MATERIAL_CONFIGS[wall]['itu_id'],
        }
        
        logger.debug(f"Sampled materials: ground={ground}, wall={wall}, rooftop={rooftop}")
        return materials
    
    def sample_properties(
        self,
        material_name: str,
    ) -> Dict[str, float]:
        """
        Sample specific property values for a material.
        
        Args:
            material_name: Material key (e.g., 'concrete')
            
        Returns:
            Dict with 'epsilon_r', 'sigma', 'freq_range_ghz'
        """
        if material_name not in self.MATERIAL_CONFIGS:
            raise ValueError(f"Unknown material: {material_name}")
        
        config = self.MATERIAL_CONFIGS[material_name]
        
        if not self.enable_randomization:
            # Return midpoint values
            return {
                'epsilon_r': np.mean(config['epsilon_r']),
                'sigma': np.mean(config['sigma']),
                'freq_range_ghz': config['freq_range_ghz'],
            }
        
        # Sample from uniform distributions
        epsilon_r = self.rng.uniform(*config['epsilon_r'])
        
        # Sample conductivity in log-space for better coverage
        sigma_min, sigma_max = config['sigma']
        if sigma_min > 0:
            log_sigma = self.rng.uniform(np.log10(sigma_min), np.log10(sigma_max))
            sigma = 10 ** log_sigma
        else:
            sigma = 0.0
        
        return {
            'epsilon_r': float(epsilon_r),
            'sigma': float(sigma),
            'freq_range_ghz': config['freq_range_ghz'],
            'category': config['category'],
        }
    
    def get_material_properties(self, itu_id: str) -> Dict[str, float]:
        """
        Get properties for an ITU material ID.
        
        Args:
            itu_id: ITU material ID (e.g., 'mat-itu_concrete')
            
        Returns:
            Material properties dictionary
        """
        # Find material by ITU ID
        for name, config in self.MATERIAL_CONFIGS.items():
            if config['itu_id'] == itu_id:
                return self.sample_properties(name)
        
        logger.warning(f"Unknown ITU material ID: {itu_id}, using default")
        return {
            'epsilon_r': 5.0,
            'sigma': 0.01,
            'freq_range_ghz': (1.0, 100.0),
        }
    
    def list_materials(self, category: Optional[str] = None) -> Dict[str, Dict]:
        """
        List available materials.
        
        Args:
            category: Filter by category ('ground', 'building', 'rooftop')
            
        Returns:
            Dictionary of material configurations
        """
        if category is None:
            return self.MATERIAL_CONFIGS
        
        return {
            name: config 
            for name, config in self.MATERIAL_CONFIGS.items() 
            if config['category'] == category
        }


if __name__ == "__main__":
    # Test material randomization
    randomizer = MaterialRandomizer(enable_randomization=True, seed=42)
    
    logger.info("Sample material combinations:")
    for i in range(5):
        materials = randomizer.sample()
        logger.info(f"\n  Combo {i+1}:")
        for surface, mat_id in materials.items():
            props = randomizer.get_material_properties(mat_id)
            logger.info(f"    {surface}: {mat_id}")
            logger.info(f"      ε_r={props['epsilon_r']:.2f}, σ={props['sigma']:.4f} S/m")
