"""
Tests for src/geo2sigmap module.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import MagicMock
sys.modules["plyfile"] = MagicMock() # Mock missing dependency

from geo2sigmap import lidar_terrain_mesh

@pytest.fixture
def mock_transformer():
    with patch("geo2sigmap.lidar_terrain_mesh.Transformer") as mock:
        yield mock


@pytest.fixture
def mock_laspy():
    with patch("geo2sigmap.lidar_terrain_mesh.laspy") as mock:
        yield mock

@pytest.fixture
def mock_pyvista():
    with patch("geo2sigmap.lidar_terrain_mesh.pv") as mock:
        yield mock

@pytest.fixture
def mock_plyfile():
    with patch("geo2sigmap.lidar_terrain_mesh.PlyData") as mock:
        yield mock

def test_generate_terrain_mesh_structure(mock_laspy, mock_pyvista, mock_plyfile, mock_transformer, tmp_path):
    """Test the structure and calls of generate_terrain_mesh."""
    
    # Mock LAS file content
    mock_las = MagicMock()
    mock_las.x = np.array([0, 10, 20])
    mock_las.y = np.array([0, 10, 20])
    mock_las.z = np.array([0, 5, 0])
    mock_las.classification = np.array([2, 2, 2]) # All ground
    mock_laspy.read.return_value = mock_las

    # Mock Transformer
    trans = MagicMock()
    mock_transformer.from_crs.return_value = trans
    # Return same coordinates (identity transform)
    trans.transform.return_value = (mock_las.x, mock_las.y)
    
    # Mock PyVista objects
    mock_cloud = MagicMock()
    mock_mesh = MagicMock()
    mock_decimated = MagicMock()
    
    # Setup chain: PolyData -> delaunay_2d -> decimate_pro
    mock_pyvista.PolyData.return_value = mock_cloud
    mock_cloud.delaunay_2d.return_value = mock_mesh
    mock_mesh.decimate_pro.return_value = mock_decimated
    
    # Setup final mesh properties for _write_ply
    mock_decimated.points = np.array([[0,0,0], [10,10,5], [20,20,0]])
    mock_decimated.faces = np.array([3, 0, 1, 2]) # 1 face
    
    input_path = "dummy.laz"
    output_path = str(tmp_path / "output.ply")
    
    # Call function
    lidar_terrain_mesh.generate_terrain_mesh(
        lidar_laz_file_path=input_path,
        ply_save_path=output_path,
        plot_figures=False
    )
    
    # Verify LAS read
    mock_laspy.read.assert_called_once_with(input_path)
    
    # Verify PyVista flow
    mock_pyvista.PolyData.assert_called()
    mock_cloud.delaunay_2d.assert_called_once()
    mock_mesh.decimate_pro.assert_called_once_with(0.90, preserve_topology=True)
    
    # Verify PLY write was triggered (PlyData initialized)
    mock_plyfile.assert_called()

def test_generate_terrain_mesh_no_ground_points(mock_laspy, mock_pyvista, mock_plyfile, mock_transformer, tmp_path):
    """Test handling when no ground points are found."""
    mock_las = MagicMock()
    mock_las.x = np.array([0, 10])
    mock_las.y = np.array([0, 10])
    mock_las.z = np.array([5, 5])
    mock_las.classification = np.array([1, 1]) # No ground (class 2)
    mock_laspy.read.return_value = mock_las

    # Mock Transformer
    trans = MagicMock()
    mock_transformer.from_crs.return_value = trans
    trans.transform.return_value = (mock_las.x, mock_las.y)
    
    mock_pyvista.PolyData.return_value = MagicMock()
    
    lidar_terrain_mesh.generate_terrain_mesh(
        "dummy.laz",
        str(tmp_path / "out.ply")
    )
    
    # Should warn and proceed with all points (implied by previous logic)
    # Just checking it doesn't crash
    mock_laspy.read.assert_called()
