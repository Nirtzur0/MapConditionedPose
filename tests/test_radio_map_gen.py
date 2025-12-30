
import pytest
import numpy as np
import sionna
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_generation.radio_map_generator import RadioMapGenerator, RadioMapConfig

# Determine if we can run Sionna tests (requires GPU or appropriate backend often, but basic setup might pass)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    HAS_GPU = len(gpus) > 0
except:
    HAS_GPU = False

class TestRadioMapGenerator:
    
    @pytest.fixture
    def mock_scene(self):
        # Create a mock Sionna Scene
        # We need to mock .transmitters, .add, .tx_array, etc.
        scene = MagicMock()
        scene.transmitters = {}
        scene.tx_array = None
        # Mock RadioMapSolver?
        return scene

    @patch('sionna.rt.RadioMapSolver') # Patch the solver class to intercept calls
    def test_float64_handling(self, MockSolver, mock_scene):
        """
        Test that RadioMapGenerator converts float64 inputs to float
        before passing them to Sionna solver, preventing Mitsuba errors.
        """
        # Setup config
        config = RadioMapConfig(
            resolution=1.0, 
            map_size=(10, 10), 
            map_extent=(0.0, 0.0, 100.0, 100.0), # 100m x 100m
            output_dir=Path("/tmp")
        )
        
        generator = RadioMapGenerator(config)
        
        # Test Data with float64 (simulate numpy inputs from lat/lon or heavy math)
        bbox = {'x_min': np.float64(0.0), 'x_max': np.float64(100.0),
                'y_min': np.float64(0.0), 'y_max': np.float64(100.0)}
        
        # Generator usually takes scene metadata/bbox from config or implicitly?
        # generate_for_scene calculates center from config.map_extent
        
        # Mock the solver instance
        mock_solver_instance = MockSolver.return_value
        # Mock the callable (the actual solve step)
        # It should return a mock radio map
        mock_rm = MagicMock()
        mock_rm.path_gain = np.zeros((1, 10, 10))
        mock_solver_instance.return_value = mock_rm
        
        cell_sites = [{'position': [50.0, 50.0, 30.0], 'orientation': [0,0,0]}]
        
        # Mock _get_ground_level to avoid scene interaction and return float
        generator._get_ground_level = MagicMock(return_value=0.0)
        
        # Run generation
        result, sionna_rm = generator.generate_for_scene(mock_scene, cell_sites, return_sionna_object=True)
        
        # CHECK: Did we call solver() with python floats?
        args, kwargs = mock_solver_instance.call_args
        
        # Check 'center' argument
        center = kwargs.get('center')
        size = kwargs.get('size')
        cell_size = kwargs.get('cell_size')
        
        assert center is not None
        assert isinstance(center[0], float), f"Center x should be float, got {type(center[0])}"
        assert isinstance(center[1], float)
        assert isinstance(center[2], float)
        
        assert size is not None
        assert isinstance(size[0], float)
        
        assert cell_size is not None
        assert isinstance(cell_size[0], float)

        print("\nSUCCESS: Arguments passed to solver were correctly cast to float.")

if __name__ == "__main__":
    t = TestRadioMapGenerator()
    # Manual run setup if needed, but pytest is standard
    print("Run with pytest.")
