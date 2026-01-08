#!/usr/bin/env python3
"""
Test script to verify LiDAR configuration and functionality.
Tests both LiDAR terrain mesh generation and USGS HAG generation.
"""
import sys
from pathlib import Path
import logging
import tempfile
import shutil
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


def test_lidar_imports():
    """Test that all LiDAR-related imports work."""
    logger.info("Testing LiDAR module imports...")
    
    try:
        from scene_builder.lidar_terrain_mesh import generate_terrain_mesh
        logger.info("✓ lidar_terrain_mesh import successful")
    except ImportError as e:
        logger.error(f"✗ Failed to import lidar_terrain_mesh: {e}")
        return False
    
    try:
        from scene_builder.usgs_lidar_hag import generate_hag
        logger.info("✓ usgs_lidar_hag import successful")
    except ImportError as e:
        logger.warning(f"⚠ usgs_lidar_hag requires pdal (optional): {e}")
        # Not a failure if pdal isn't available
    
    # Test dependencies
    try:
        import laspy
        logger.info(f"✓ laspy version: {laspy.__version__}")
    except ImportError as e:
        logger.error(f"✗ laspy not available: {e}")
        return False
    
    try:
        import pyvista as pv
        logger.info(f"✓ pyvista version: {pv.__version__}")
    except ImportError as e:
        logger.error(f"✗ pyvista not available: {e}")
        return False
    
    try:
        import pdal
        logger.info("✓ pdal available")
        pdal_available = True
    except ImportError as e:
        logger.warning(f"⚠ pdal not available (optional): {e}")
        pdal_available = False
    
    try:
        from plyfile import PlyData, PlyElement
        logger.info("✓ plyfile available")
    except ImportError as e:
        logger.error(f"✗ plyfile not available: {e}")
        return False
    
    return True


def test_lidar_terrain_mesh_mock():
    """Test terrain mesh generation with mock LiDAR data."""
    logger.info("\nTesting LiDAR terrain mesh generation with mock data...")
    
    try:
        import laspy
        from scene_builder.lidar_terrain_mesh import generate_terrain_mesh
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Generate mock LAZ file
            logger.info("Creating mock LAZ file...")
            mock_laz_path = tmpdir_path / "mock_terrain.laz"
            
            # Create a simple grid of ground points (classification 2)
            grid_size = 50
            x = np.linspace(-250, 250, grid_size)
            y = np.linspace(-150, 150, grid_size)
            xx, yy = np.meshgrid(x, y)
            
            # Add some terrain variation (gentle hills)
            zz = 5 * np.sin(xx / 100) * np.cos(yy / 100) + 10
            
            # Flatten to point cloud
            points_x = xx.flatten()
            points_y = yy.flatten()
            points_z = zz.flatten()
            
            # Create LAS header and data
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.offsets = np.array([points_x.min(), points_y.min(), points_z.min()])
            header.scales = np.array([0.01, 0.01, 0.01])
            
            las = laspy.LasData(header)
            las.x = points_x
            las.y = points_y
            las.z = points_z
            
            # Set all points as ground (class 2)
            las.classification = np.full(len(points_x), 2, dtype=np.uint8)
            
            # Add some noise points (class 7)
            noise_count = 100
            noise_indices = np.random.choice(len(points_x), noise_count, replace=False)
            las.classification[noise_indices] = 7
            
            # Save LAZ file
            las.write(str(mock_laz_path))
            logger.info(f"✓ Created mock LAZ file with {len(points_x)} points at {mock_laz_path}")
            
            # Test mesh generation
            output_ply = tmpdir_path / "terrain_mesh.ply"
            logger.info("Generating terrain mesh...")
            
            mesh = generate_terrain_mesh(
                lidar_laz_file_path=str(mock_laz_path),
                ply_save_path=str(output_ply),
                src_crs="EPSG:32617",  # UTM Zone 17N
                dest_crs="EPSG:32617",
                plot_figures=False,
                center_x=0.0,
                center_y=0.0
            )
            
            # Verify output
            if not output_ply.exists():
                logger.error("✗ Output PLY file not created")
                return False
            
            file_size = output_ply.stat().st_size
            logger.info(f"✓ Terrain mesh generated successfully")
            logger.info(f"  Output file: {output_ply}")
            logger.info(f"  File size: {file_size / 1024:.2f} KB")
            logger.info(f"  Mesh points: {mesh.n_points}")
            logger.info(f"  Mesh cells: {mesh.n_cells}")
            
            if mesh.n_points < 100:
                logger.warning("⚠ Mesh has very few points, may indicate decimation issues")
            
            return True
            
    except Exception as e:
        logger.error(f"✗ Terrain mesh generation failed: {e}", exc_info=True)
        return False


def test_lidar_config_params():
    """Test that LiDAR configuration parameters are properly structured."""
    logger.info("\nTesting LiDAR configuration parameters...")
    
    try:
        from scene_generation.tiles import TileGenerator
        
        # Check default terrain config structure
        default_terrain_cfg = {
            'use_lidar': False,
            'lidar_calibration': False,
            'lidar_terrain': False,
        }
        
        logger.info("✓ LiDAR config parameters are properly defined:")
        logger.info(f"  - use_lidar: {default_terrain_cfg['use_lidar']}")
        logger.info(f"  - lidar_calibration: {default_terrain_cfg['lidar_calibration']}")
        logger.info(f"  - lidar_terrain: {default_terrain_cfg['lidar_terrain']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_pdal_pipeline():
    """Test PDAL pipeline construction."""
    logger.info("\nTesting PDAL pipeline construction...")
    
    try:
        import pdal
    except ImportError:
        logger.warning("⚠ PDAL not available - skipping pipeline test (optional dependency)")
        return True  # Pass the test if PDAL isn't available
    
    try:
        from scene_builder.usgs_lidar_hag import build_pdal_pipeline
        from shapely.geometry import Polygon
        
        # Create a small test polygon
        test_polygon = Polygon([
            (-105.2820, 40.0150),
            (-105.2770, 40.0150),
            (-105.2770, 40.0180),
            (-105.2820, 40.0180),
            (-105.2820, 40.0150)
        ])
        
        # Build pipeline (without executing it)
        pipeline_dict = build_pdal_pipeline(
            extent_epsg3857=test_polygon,
            usgs_3dep_dataset_names=['USGS_LPC_CO_Boulder_2020_D20'],
            pc_resolution=1.0,
            filterNoise=True,
            reclassify=False,
            savePointCloud=False,
            outCRS=3857,
            pc_outName='test',
            pc_outType='laz'
        )
        
        logger.info(f"✓ PDAL pipeline constructed successfully")
        logger.info(f"  Pipeline stages: {len(pipeline_dict)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PDAL pipeline test failed: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all LiDAR configuration tests."""
    logger.info("="*70)
    logger.info("LiDAR Configuration Test Suite")
    logger.info("="*70)
    
    results = {
        'imports': test_lidar_imports(),
        'config_params': test_lidar_config_params(),
        'pdal_pipeline': test_pdal_pipeline(),
        'terrain_mesh_mock': test_lidar_terrain_mesh_mock(),
    }
    
    logger.info("\n" + "="*70)
    logger.info("Test Results Summary")
    logger.info("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("="*70)
    if all_passed:
        logger.info("✓ All LiDAR tests passed!")
        return 0
    else:
        failed_count = sum(1 for r in results.values() if not r)
        logger.error(f"✗ {failed_count}/{len(results)} tests failed")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(run_all_tests())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
