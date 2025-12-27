
import numpy as np
import logging
from typing import Tuple, Dict, Any, List
import sionna as sn

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from skimage import draw
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class OSMRasterizer:

    """
    Rasterize Sionna scene objects into multi-channel OSM maps.
    
    Generates 5-channel maps:
    0: Height (meters)
    1: Material ID (integer)
    2: Footprint (binary)
    3: Road (binary)
    4: Terrain (binary)
    """
    
    # Standard material mapping (can be extended)
    MATERIAL_MAP = {
        'itu_concrete': 1,
        'itu_brick': 2,
        'itu_glass': 3,
        'itu_metal': 4,
        'itu_wood': 5,
        'itu_marble': 6,
        'itu_plasterboard': 7,
        'itu_plywood': 8,
        'itu_chipboard': 9,
        'itu_floorboard': 10,
        'itu_ceiling_board': 11,
        'itu_wet_ground': 20,
        'itu_dry_ground': 21,
    }

    def __init__(
        self,
        map_size: Tuple[int, int] = (512, 512),
        map_extent: Tuple[float, float, float, float] = (0.0, 0.0, 512.0, 512.0),
    ):
        """
        Args:
            map_size: (width, height) in pixels
            map_extent: (x_min, y_min, x_max, y_max) in meters
        """
        self.width, self.height = map_size
        self.x_min, self.y_min, self.x_max, self.y_max = map_extent
        self.resolution_x = (self.x_max - self.x_min) / self.width
        self.resolution_y = (self.y_max - self.y_min) / self.height
        
        self.scale = np.array([1.0 / self.resolution_x, 1.0 / self.resolution_y])
        self.offset = np.array([-self.x_min, -self.y_min])
        
        if not (CV2_AVAILABLE or SKIMAGE_AVAILABLE):
            logger.warning("Neither OpenCV nor Scikit-Image available. Rasterization will be slow or fail.")

    def _world_to_pixel(self, points_2d: np.ndarray) -> np.ndarray:
        """Convert world coordinates (x, y) to pixel coordinates (col, row)."""
        # (x, y) -> (x - x_min, y - y_min) -> scale -> pixel
        # Note: Image origin is usually top-left. 
        # For maps, we often want bottom-left or consistent with RadioMap.
        # RadioMapGenerator usually does np.meshgrid(x, y) where x is col, y is row (or vice versa).
        # Let's assume math convention: y increases upwards. 
        # But images have y increasing downwards.
        # To match Sionna RadioMap (which is likely x-y grid), we should check.
        # Sionna RadioMapSolver output is [num_tx, num_rx_y, num_rx_x]? 
        # Or [num_tx, size_y, size_x]?
        # Usually it matches the meshgrid indexing.
        
        pixels = (points_2d + self.offset) * self.scale
        # If we need to flip Y for image coords:
        # pixels[:, 1] = self.height - pixels[:, 1]
        # But let's stick to simple scaling first and verify alignment later.
        
        return pixels.astype(np.int32)

    def rasterize(self, scene) -> np.ndarray:
        """
        Rasterize the scene into a 5-channel map.
        
        Returns:
            map: (5, height, width) np.float32 array
        """
        # Channels: Height, Material, Footprint, Road, Terrain
        osm_map = np.zeros((5, self.height, self.width), dtype=np.float32)
        
        # Iterating objects
        for name, obj in scene.objects.items():
            if not hasattr(obj, 'mi_mesh'):
                continue
                
            # Heuristics for classification
            is_ground = 'ground' in name.lower() or 'terrain' in name.lower()
            is_road = 'road' in name.lower() or 'street' in name.lower()
            
            # Get geometry
            try:
                # Mitsuba 3 / DrJit mesh access
                # For basic vertex access we might need to drill down
                # obj.mi_mesh is a mitsuba.Mesh
                # We can access vertex_positions_buffer
                
                # Check for vertex_positions attribute or similar (DrJit tensor)
                # We need to convert to numpy
                trimesh = obj.mi_mesh
                
                # Get vertices (float array) and faces (int array)
                # Depending on version this might differ. 
                # Assuming .vertex_positions_buffer() returns flat array or similar
                # Let's use simple property if available
                # Or DrJit copy
                import drjit as dr
                vertices = np.array(trimesh.vertex_positions_buffer()) # Flat buffer?
                faces = np.array(trimesh.faces_buffer()) # Flat buffer?
                
                # Reshape if flat
                if len(vertices.shape) == 1:
                    vertices = vertices.reshape(-1, 3)
                if len(faces.shape) == 1:
                    faces = faces.reshape(-1, 3)
                
            except Exception as e:
                logger.debug(f"Failed to extract mesh for {name}: {e}")
                continue
            
            # Extract Z height
            z_values = vertices[:, 2]
            
            # Project vertices to 2D
            verts_2d = vertices[:, :2]
            pixels = self._world_to_pixel(verts_2d)
            
            # Debug
            # print(f"Object: {name}")
            # print(f"  Vertices: {vertices.shape}, Range: {np.min(vertices, axis=0)} to {np.max(vertices, axis=0)}")
            # print(f"  Pixels: {np.min(pixels, axis=0)} to {np.max(pixels, axis=0)}")
            
            # Prepare material ID

            mat_id = 0
            if hasattr(obj, 'radio_material'):
                mat_name = obj.radio_material.name
                mat_id = self.MATERIAL_MAP.get(mat_name, 0)
                # Fallback for generic types
                if mat_id == 0:
                    if 'concrete' in mat_name: mat_id = 1
                    elif 'metal' in mat_name: mat_id = 4
                    elif 'wood' in mat_name: mat_id = 5
                    elif 'ground' in mat_name: mat_id = 20
            
            # Rasterize triangles
            if CV2_AVAILABLE:
                self._rasterize_cv2(osm_map, pixels, faces, z_values, mat_id, is_ground, is_road)
            elif PIL_AVAILABLE:
                self._rasterize_pil(osm_map, pixels, faces, z_values, mat_id, is_ground, is_road)
            elif SKIMAGE_AVAILABLE:
                self._rasterize_skimage(osm_map, pixels, faces, z_values, mat_id, is_ground, is_road)
            else:
                logger.warning("No rasterization backend available.")

        return osm_map

    def _rasterize_pil(self, osm_map, pixels, faces, z_values, mat_id, is_ground, is_road):
        # PIL rasterization
        # Iterate triangles
        # Note: Triangles are defined by faces pointing to pixels
        # Create temp images for masks if needed, or draw directly
        
        # Optimization: Draw all triangles for a feature at once?
        # ImageDraw.polygon takes one polygon.
        # So we iterate.
        
        # Prepare "canvases" for features being updated
        # We can draw onto temporary PIL images and then max-composite into numpy
        # Creating a PIL image for every object might be slow.
        # Better: Create one PIL image per channel for the whole scene?
        # But we are inside object loop. OSMRasterizer.rasterize iterates objects.
        # We modify osm_map directly (numpy).
        
        # For this object, we can create a mask.
        mask_img = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        # Collect triangles
        # triangles = pixels[faces] # (N, 3, 2)
        # Convert to list of tuples for PIL
        # This python loop might be slow for large meshes.
        
        for face in faces:
             pts = pixels[face]
             # Flatten to [(x1,y1), (x2,y2), (x3,y3)]
             poly = [tuple(pts[0]), tuple(pts[1]), tuple(pts[2])]
             draw.polygon(poly, fill=1, outline=1)
             
        mask = np.array(mask_img)
        non_zero = mask > 0
        
        # Update channels
        # Height (Channel 0) - Max Z
        if not is_ground:
            max_z = np.max(z_values) # Object max
            osm_map[0][non_zero] = np.maximum(osm_map[0][non_zero], max_z)
            
        # Material (Channel 1)
        osm_map[1][non_zero] = mat_id
        
        # Footprint (Channel 2)
        if not is_ground:
            osm_map[2][non_zero] = 1.0
            
        # Road (Channel 3)
        if is_road:
            osm_map[3][non_zero] = 1.0
            
        # Terrain (Channel 4)
        if is_ground:
            osm_map[4][non_zero] = 1.0



    def _rasterize_cv2(self, osm_map, pixels, faces, z_values, mat_id, is_ground, is_road):
        # We need to draw triangles.
        # CV2 fillPoly or drawContours
        # Faces are indices into pixels
        
        # Get triangles: (N, 3, 2)
        tris = pixels[faces] 
        
        # 1. Footprint & Classification (Road, Terrain)
        # Use simple fillPoly with max
        
        # We process separately to handle height (max)
        # For binary masks, we can draw all at once
        
        points_list = [t.reshape((-1, 1, 2)) for t in tris]
        
        # Footprint (Channel 2)
        if not is_ground: # Ground is usually full coverage, we might treat it as footprint or terrain
            # If it's a building, it's a footprint
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, points_list, 1)
            osm_map[2] = np.maximum(osm_map[2], mask)
        
        # Terrain (Channel 4)
        if is_ground:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, points_list, 1)
            osm_map[4] = np.maximum(osm_map[4], mask)
            
        # Road (Channel 3)
        if is_road:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, points_list, 1)
            osm_map[3] = np.maximum(osm_map[3], mask)
            
        # Material (Channel 1)
        # We overwrite material (or take max? usually overlap means problem or z-buffer)
        # We simply paint material ID
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, points_list, int(mat_id))
        # Update map where mask > 0 AND new material is valid
        # Simple overwrite logic
        non_zero = mask > 0
        osm_map[1][non_zero] = mask[non_zero]

        # Height (Channel 0)
        # Use per-triangle max Z for better height variation
        # Each triangle gets the max Z of its 3 vertices
        if not is_ground:
            # Calculate max Z per triangle for better detail
            tri_z_max = z_values[faces].max(axis=1)  # (N,) - max Z per triangle
            
            # Group triangles by quantized height for efficiency
            z_quantized = np.round(tri_z_max).astype(np.int32)
            unique_heights = np.unique(z_quantized)
            
            # Rasterize triangles grouped by similar height
            for height_val in unique_heights:
                height_mask = z_quantized == height_val
                height_tris = tris[height_mask]
                if len(height_tris) == 0:
                    continue
                    
                height_points = [t.reshape((-1, 1, 2)) for t in height_tris]
                temp_mask = np.zeros((self.height, self.width), dtype=np.float32)
                cv2.fillPoly(temp_mask, height_points, float(height_val))
                
                # Use max to handle overlapping triangles
                osm_map[0] = np.maximum(osm_map[0], temp_mask)
        else:
            # For ground, use actual Z (usually 0 or close to it)
            pass

    def _rasterize_skimage(self, osm_map, pixels, faces, z_values, mat_id, is_ground, is_road):
        # Fallback implementation
        pass

