"""
LiDAR Terrain Mesh Generator

This module handles the conversion of LiDAR point cloud data (.laz) into 3D terrain meshes (.ply).
It filters ground points, transforms coordinates, and generates a triangulated mesh.
"""

import os
import logging
import laspy
import numpy as np
import pyvista as pv
from pyproj import Transformer
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)

def generate_terrain_mesh(
    lidar_laz_file_path: str,
    ply_save_path: str,
    src_crs: str = "EPSG:3857",
    dest_crs: str = "EPSG:32617",
    plot_figures: bool = False,
    center_x: float = 0.0,
    center_y: float = 0.0
) -> pv.PolyData:
    """
    Generate a terrain mesh from a LiDAR .laz file.

    Args:
        lidar_laz_file_path (str): Path to input .laz file.
        ply_save_path (str): Path to output .ply file.
        src_crs (str): Source Coordinate Reference System (default: EPSG:3857).
        dest_crs (str): Destination Coordinate Reference System.
        plot_figures (bool): Whether to visualize the mesh after generation.
        center_x (float): X-coordinate center offset.
        center_y (float): Y-coordinate center offset.

    Returns:
        pv.PolyData: The generated terrain mesh.
    """
    logger.info(f"Generating terrain mesh from {lidar_laz_file_path}")
    
    # Configure PyVista backend for headless servers
    pv.global_theme.trame.server_proxy_enabled = True

    try:
        # Load the LAZ file
        las = laspy.read(lidar_laz_file_path)
    except Exception as e:
        logger.error(f"Failed to read LAZ file {lidar_laz_file_path}: {e}")
        raise

    # Extract the classification field and filter for ground points (classification == 2)
    ground_mask = las.classification == 2
    if not np.any(ground_mask):
        logger.warning("No ground points (class 2) found in LiDAR data. Using all points.")
        ground_mask = np.ones(len(las.x), dtype=bool)

    # Extract coordinates
    x_ground = las.x[ground_mask]
    y_ground = las.y[ground_mask]
    z_ground = las.z[ground_mask]

    # Transform coordinates to destination CRS
    transformer = Transformer.from_crs(src_crs, dest_crs, always_xy=True)
    x_ground, y_ground = transformer.transform(x_ground, y_ground)

    # Calculate Z-offset relative to center point using KD-Tree
    points_2d = np.vstack((x_ground, y_ground)).T
    tree = cKDTree(points_2d)
    
    # Query the nearest point to the target center coordinates
    # Note: center_x/y should already be in dest_crs or this query might be mismatched if mixed
    _, index = tree.query([center_x, center_y])
    
    nearest_z = z_ground[index]
    logger.info(f"Nearest Z value at center ({center_x}, {center_y}): {nearest_z}")

    # Normalize Z-values
    z_ground = z_ground - nearest_z

    # Center XY coordinates
    x_ground = x_ground - center_x
    y_ground = y_ground - center_y
    logger.info(f"Centered coordinates. Bounds X: [{x_ground.min():.2f}, {x_ground.max():.2f}], Y: [{y_ground.min():.2f}, {y_ground.max():.2f}]")

    # Create PyVista Point Cloud
    points = np.vstack((x_ground, y_ground, z_ground)).T
    point_cloud = pv.PolyData(points)

    # Create Surface Mesh via Delaunay 2D Triangulation
    surface_mesh = point_cloud.delaunay_2d()
    logger.info(f"Original mesh faces: {surface_mesh.n_faces}")

    # Decimate mesh to reduce complexity (90% reduction)
    pro_decimated = surface_mesh.decimate_pro(0.90, preserve_topology=True)
    logger.info(f"Decimated mesh faces: {pro_decimated.n_faces}")
    
    # Use the decimated mesh
    final_mesh = pro_decimated

    # Export to PLY
    _write_ply(final_mesh, ply_save_path)
    logger.info(f"Saved terrain mesh to {ply_save_path}")

    # Visualization
    if plot_figures:
        _plot_mesh(final_mesh)

    return final_mesh

def _write_ply(mesh: pv.PolyData, path: str):
    """Refactored helper to write PLY file manually using plyfile."""
    vertices = mesh.points
    # Faces shape is (N, 4) where col 0 is num_vertices (3 for triangles), col 1-3 are indices
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]

    vertex_data = [(v[0], v[1], v[2]) for v in vertices]
    face_data = [(list(face),) for face in faces]

    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex'
    )
    face_element = PlyElement.describe(
        np.array(face_data, dtype=[('vertex_indices', 'i4', (3,))]), 'face'
    )

    PlyData([vertex_element, face_element], text=False).write(path)

def _plot_mesh(mesh: pv.PolyData):
    pv.set_jupyter_backend('client')
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=mesh.points[:, 2], cmap="terrain")
    plotter.show()

if __name__ == '__main__':
    # Configuration
    DATA_DIR = "output"
    MESH_DATA_DIR = os.path.join(DATA_DIR, "mesh")
    os.makedirs(MESH_DATA_DIR, exist_ok=True)
    
    CENTER_X = -71.0602
    CENTER_Y = 42.3512
    PROJ_EPSG = "EPSG:32613" # UTM Zone 19N for Boston/NYC area approx?
    
    input_laz = os.path.join(DATA_DIR, "test_hag.laz")
    output_ply = os.path.join(MESH_DATA_DIR, "lidar_terrain.ply")
    
    if os.path.exists(input_laz):
        generate_terrain_mesh(
            lidar_laz_file_path=input_laz,
            ply_save_path=output_ply,
            src_crs="EPSG:3857",
            dest_crs=PROJ_EPSG,
            plot_figures=False,
            center_x=CENTER_X,
            center_y=CENTER_Y
        )
    else:
        logger.warning(f"Input file not found: {input_laz}")