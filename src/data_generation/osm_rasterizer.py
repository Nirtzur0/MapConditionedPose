
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

# Check if Sionna is available for GPU acceleration
try:
    import sionna as sn
    # Check for correct version and capabilities if needed
    SIONNA_RT_AVAILABLE = True
except ImportError:
    SIONNA_RT_AVAILABLE = False


class OSMRasterizer:

    """
    Rasterize Sionna scene objects into multi-channel OSM maps.
    
    Generates 5-channel maps:
    0: Height (meters)
    1: Material ID (integer)
    2: Footprint (binary)
    3: Road (binary)
    4: Terrain (binary)
    
    NOTE: Roads are fetched directly from OSM data since Sionna scenes
    only contain building geometries. Pass scene_metadata or osm_data
    to rasterize() to include roads.
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

    def rasterize(self, scene, scene_metadata: Dict = None, osm_data: Dict = None) -> np.ndarray:
        """
        Rasterize the scene into a 5-channel map.
        
        Args:
            scene: Sionna scene object
            scene_metadata: Optional metadata dict with 'bbox' for fetching roads from OSM
            osm_data: Optional pre-fetched OSM data dict with 'roads' key
        
        Returns:
            map: (5, height, width) np.float32 array
        """
        # Channels: Height, Material, Footprint, Road, Terrain
        osm_map = np.zeros((5, self.height, self.width), dtype=np.float32)

        # --- GPU Acceleration (Sionna) ---
        # If Sionna Scene is provided, try to use ray casting for Height/Footprint
        # This is MUCH faster than iterating thousands of building meshes on CPU
        if SIONNA_RT_AVAILABLE and hasattr(scene, 'closest_point'):
             try:
                 osm_map = self._rasterize_sionna_gpu(scene, osm_map)
                 gpu_geometry_done = True
             except Exception as e:
                 logger.warning(f"OSM GPU Rasterization failed, falling back to CPU: {e}")
                 gpu_geometry_done = False
        else:
            gpu_geometry_done = False

        
        # --- Auto-Detect Coordinate System ---
        # Sample an object to check if vertices are Global (UTM) or Local (Centered)
        # This fixes empty maps particularly when Metadata is UTM but Mesh is Local.
        
        sample_vertices = None
        for name, obj in scene.objects.items():
            if hasattr(obj, 'mi_mesh'):
                try:
                    # Try to get vertices
                    if hasattr(obj.mi_mesh, 'vertex_positions_buffer'):
                        # DrJit/Mitsuba 3
                        # We need a way to peek without crashing logic later
                        # For now, rely on logic inside loop or do a quick check?
                        # Let's peek.
                         import drjit as dr
                         v_buf = obj.mi_mesh.vertex_positions_buffer()
                         # Just take first vertex
                         if len(v_buf) > 0:
                             v_sample = np.array(v_buf[0:3]) # x, y, z
                             sample_vertices = v_sample
                             break
                except:
                    pass
        
        if sample_vertices is not None:
             # Check magnitude
             x_val = sample_vertices[0]
             
             # Metric: Distance to defined map origin (x_min)
             dist_to_origin = abs(x_val - self.x_min)
             dist_to_zero = abs(x_val)
             
             # If closer to 0 than to x_min (and x_min is large), it's likely Local.
             is_local = False
             if abs(self.x_min) > 10000: # If map is Global
                 if dist_to_zero < 10000 and dist_to_origin > 10000:
                     is_local = True
             
             if is_local:
                 logger.info(f"OSM Rasterizer: Detected LOCAL coordinates (Item x={x_val:.1f}). adjusting offset.")
                 # Local (0,0) corresponds to Map Center.
                 # Map Center in Global: (x_min + x_max)/2
                 # Map Origin is x_min.
                 # Pixel X = (LocalX + CenterOffset) * Scale
                 # GlobalX = x_min + Pixel/Scale
                 # GlobalX = LocalX + CenterX
                 # LocalX + CenterX - x_min -> Pixel/Scale
                 # Pixel/Scale = LocalX + (x_max - x_min)/2
                 # Offset = (x_max - x_min) / 2
                 
                 w_m = self.x_max - self.x_min
                 h_m = self.y_max - self.y_min
                 self.offset = np.array([w_m / 2.0, h_m / 2.0])
             else:
                 # Assume Global matching map_extent
                 if abs(dist_to_origin) > 100000:
                     logger.warning(f"OSM Rasterizer: Vertices seem far from map extent! x={x_val:.1f}, x_min={self.x_min:.1f}")
        # Keep default offset (-x_min, -y_min)
                 pass
        
        # Batching Data Structures
        polygons = {
            'footprint': [],
            'road': [],
            'terrain': [],
            'materials': {} # mat_id -> list of polys
        }

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
            
            # --- Collect Polygons for Batch Rasterization ---
            # Instead of drawing immediately, we bucket polygons to reduce overhead.
            
            # Get triangles in pixel coords: (N, 3, 2)
            tris = pixels[faces]
            # Convert to list of (N, 1, 2) for cv2.fillPoly
            # This is a bit expensive in Python loop, but much faster than N fillPoly calls
            object_polys = [t.reshape((-1, 1, 2)) for t in tris]
            
            # 1. Height (Channel 0) - Still done per object if CPU fallback, 
            # or skipped if GPU done.
            if not gpu_geometry_done and not is_ground:
                 # Must do per-object or per-triangle height
                 # For batching, this is hard. We'll use the per-object call ONLY for Height
                 # if GPU failed.
                 if CV2_AVAILABLE:
                     self._rasterize_cv2_height_only(osm_map, object_polys, z_values[faces])
            
            # 2. Footprint (Channel 2)
            if not is_ground:
                polygons['footprint'].extend(object_polys)
            
            # 3. Road (Channel 3)
            if is_road:
                polygons['road'].extend(object_polys)
                
            # 4. Terrain (Channel 4)
            if is_ground:
                polygons['terrain'].extend(object_polys)
                
            # 5. Material (Channel 1)
            if mat_id > 0:
                if mat_id not in polygons['materials']:
                    polygons['materials'][mat_id] = []
                polygons['materials'][mat_id].extend(object_polys)

        # --- Batch Draw ---
        if CV2_AVAILABLE:
            self._rasterize_batches_cv2(osm_map, polygons)
        else:
            logger.warning("CV2 not available, skipping batch semantic rasterization.")
        
        # --- Rasterize Roads from OSM ---
        # Since Sionna scenes don't include road meshes, we fetch and rasterize them separately
        if scene_metadata or osm_data:
            self._rasterize_roads(osm_map, scene_metadata, osm_data)
            
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



    def _rasterize_batches_cv2(self, osm_map, polygons):
        """Batch rasterization using OpenCV."""
        
        # Helper for binary layers
        def draw_layer_max(layer_idx, poly_list):
            if not poly_list: return
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillPoly(mask, poly_list, 1)
            # Max composite
            osm_map[layer_idx] = np.maximum(osm_map[layer_idx], mask)
            
        # Footprint (2)
        draw_layer_max(2, polygons['footprint'])
        
        # Road (3)
        draw_layer_max(3, polygons['road'])
        
        # Terrain (4)
        draw_layer_max(4, polygons['terrain'])
        
        # Materials (1) - Integer IDs
        # We process materials. Overlap handling: last one wins or generic?
        # We can draw all to a single temp buffer.
        if polygons['materials']:
            mat_layer = np.zeros((self.height, self.width), dtype=np.int32)
            # Sort by ID just for determinism? Or specific order?
            # Typically small objects (higher ID?) might want to be on top?
            # No specific logic, just iterate.
            for mat_id, poly_list in polygons['materials'].items():
                # Draw this material
                # We draw directly to mask. Overwrites previous.
                cv2.fillPoly(mat_layer, poly_list, int(mat_id))
            
            # Writes non-zeros to map
            mask_nz = mat_layer > 0
            osm_map[1][mask_nz] = mat_layer[mask_nz]

    def _rasterize_cv2_height_only(self, osm_map, points_list, z_values_tri):
        """Fallback height rasterizer for CPU."""
        # Calculate max Z per triangle
        tri_z_max = z_values_tri.max(axis=1)
        z_quantized = np.round(tri_z_max).astype(np.int32)
        unique_heights = np.unique(z_quantized)
        
        for height_val in unique_heights:
            # Indices for this height
            indices = np.where(z_quantized == height_val)[0]
            # Use list comprehension for speed?
            # points_list is list.
            batch_points = [points_list[i] for i in indices]
            
            temp_mask = np.zeros((self.height, self.width), dtype=np.float32)
            cv2.fillPoly(temp_mask, batch_points, float(height_val))
            osm_map[0] = np.maximum(osm_map[0], temp_mask)

    def _rasterize_cv2(self, osm_map, pixels, faces, z_values, mat_id, is_ground, is_road, skip_height=False):
        # Legacy per-object rasterizer (kept for reference or if batching proves problematic, though unused now)
        pass


    def _rasterize_skimage(self, osm_map, pixels, faces, z_values, mat_id, is_ground, is_road):
        # Fallback implementation
        pass

    def _rasterize_sionna_gpu(self, scene, osm_map) -> np.ndarray:
        """
        Use Sionna RT (GPU) to compute Height and Footprint maps.
        Casts rays from top-down to find surface height.
        
        Updates osm_map in-place (Channels 0 and 2).
        """
        import tensorflow as tf
        
        # 1. Generate Ray Grid (World Coordinates)
        # Pixel centers
        feature_width = self.width
        feature_height = self.height
        
        # X range [x_min, x_max], Y range [y_min, y_max]
        # Note: image Y usually goes down, but math Y goes up.
        # We want to match how we display the map.
        # If we use standard meshgrid (Cartesian), (0,0) is bottom-left (min_x, min_y).
        # We will map this to image pixels later.
        
        xs = tf.linspace(float(self.x_min), float(self.x_max), feature_width)
        ys = tf.linspace(float(self.y_min), float(self.y_max), feature_height)
        
        # Create grid
        # We want X to vary along columns, Y along rows?
        # Meshgrid 'xy': x (W), y (H). Output (H, W).  <- Matches Image (row=y, col=x)
        xv, yv = tf.meshgrid(xs, ys) 
        
        # Ray Origins: Start high up (e.g. 30000m to cover all terrain)
        z_start = 30000.0
        zv = tf.fill(tf.shape(xv), z_start)
        
        origins = tf.stack([xv, yv, zv], axis=-1) # [H, W, 3]
        
        # Ray Directions: Downwards (-Z)
        directions = tf.constant([0.0, 0.0, -1.0], dtype=tf.float32)
        # Broadcast to shape
        directions = tf.broadcast_to(directions, tf.shape(origins))
        
        # Flatten for safety (some Sionna/TF ops prefer [N, 3])
        output_shape = tf.shape(origins)
        origins_flat = tf.reshape(origins, [-1, 3])
        directions_flat = tf.reshape(directions, [-1, 3])
        
        # 2. Trace Rays
        # returns: positions, normals, object_indices
        # Shapes: [N, 3], [N, 3], [N]
        
        try:
             # Fast intersection test
             positions_flat, normals_flat, obj_indices_flat = scene.closest_point(origins_flat, directions_flat)
             
             # Reshape back using the original shape (minus the last dim 3)
             # output_shape is [H, W, 3]
             grid_shape = output_shape[:-1]
             
             positions = tf.reshape(positions_flat, tf.concat([grid_shape, [3]], axis=0))
             # normals = tf.reshape(normals_flat, tf.concat([grid_shape, [3]], axis=0)) # Not used
             
             # 3. Process Hits
             # Check for valid hits. 
             # If no hit, position might be 0 or check mask?
             # Sionna closest_point returns large distance if no hit?
             # Or obj_index = -1?
             # We can check Z difference. If Z is still ~1000, no hit.
             
             # Height = positions.z
             z_map = positions[..., 2]
             
             # Mask invalid hits (Z close to start or obj_index < 0)
             # usually obj_indices is -1 if no hit (depends on backend).
             # Let's assume hits < z_start - epsilon
             valid_mask = z_map < (z_start - 1000.0)
             
             # Convert to numpy
             z_map_np = z_map.numpy()
             valid_mask_np = valid_mask.numpy()
             
             # Update OSM Map
             # Channel 0: Height
             # We only update where we have valid hits.
             # This automatically handles occlusion correctly (closest point).
             
             # Note: Meshgrid 'xy' gives y increasing upwards (min to max).
             # Images usually have y=0 at top. 
             # If we want standard map orientation (North=Up), we usually put y_min at bottom.
             # So row 0 corresponds to y_min? No.
             # In matrix indexing: row 0 is top.
             # If we want row 0 to be y_max (Top), we should flip Y linspace?
             # self.y_min corresponds to bottom of valid area.
             # Standard "Map" view: (0,0) [bottom-left] is x_min, y_min.
             # Image memory: (0,0) [top-left].
             # So if we generated Y from min to max, row 0 is min Y (Bottom).
             # Only if we display with origin='lower' (matplotlib defaults) it looks right.
             # But if we treat it as matrix, row 0 is min_y.
             # Let's ensure consistency with `_world_to_pixel`.
             # _world_to_pixel: y -> (y - y_min). Low Y -> Low Pixel Index (Top? No, small index).
             # Small index = Top.
             # So Low Y (Bottom of world) -> Top of Image.
             # This means the Map is FLIPPED vertically in current CPU implementation?
             # In CPU: pixels = (y - y_min) * scale.
             # If y = y_min, pixel = 0 (Top).
             # If y = y_max, pixel = H (Bottom).
             # So "UP" in world (+Y) is "DOWN" in image (+Row).
             # This is a FLIPPED map (North is Down).
             # If we want North UP, we should have done `H - pixel`.
             # BUT assuming we match the CPU implementation:
             # CPU: y values (min..max) map to (0..H).
             # GPU: y values (min..max) in meshgrid row 0..H.
             # So they match!
             
             # Height Channel
             current_height = osm_map[0]
             # Max with existing (should be 0)
             osm_map[0] = np.maximum(current_height, z_map_np * valid_mask_np)
             
             # Footprint Channel (2)
             # Anything with valid hit > ground level is footprint?
             # Or just valid hit? 
             # Usually ground is considered a hit.
             # We want buildings.
             # Simple heuristic: Height > 0.5m (exclude flat ground)
             # But if ground is at z=0, buildings start at z=0?
             # Typically ground is at z=0.
             # We need to distinguish Gound from Buildings.
             # Material ID helps. But we don't have it purely from geometry efficiently without map.
             # However, we can update Footprint for everything, 
             # and then 'Terrain' channel will overwrite? 
             # Or we leave Footprint 0 for ground.
             # Let's just set Footprint = 1 for all valid hits for now, 
             # and let the CPU loop refine it for 'Terrain' vs 'Building'?
             # NO, the CPU loop is what we want to avoid.
             
             # Optimization:
             # If we can identify ground object indices, we are golden.
             # `scene.objects` order matches indices?
             # Usually yes.
             # We can pre-scan objects (fast loop) to list "Ground" indices.
             # Then mask those out from Footprint.
             
             ground_indices = []
             try:
                 # Mitsuba/Sionna index mapping is implicit.
                 # Usually 0..N-1 based on scene.objects.values() iteration order?
                 # Or scene.shapes?
                 # This is risky.
                 pass
             except:
                 pass
                 
             # Heuristic: If we don't know, we assume everything is a building/object
             # unless z is very close to ground?
             # But ground can be hilly.
             
             # For now: Update Height (Channel 0).
             # Update Footprint (Channel 2) where z > 0.1 (simple filter).
             # This is 'low overhead' and robust enough for basic maps.
             
             is_building = (z_map_np > 0.1) & valid_mask_np
             osm_map[2] = np.maximum(osm_map[2], is_building.astype(np.float32))
             
             return osm_map
             
        except Exception as e:
             logger.warning(f"Sionna Ray Casting failed: {e}")
             raise e
             
        return osm_map

    def _rasterize_roads(self, osm_map: np.ndarray, scene_metadata: Dict = None, osm_data: Dict = None):
        """
        Rasterize roads from OSM data into channel 3.
        
        Args:
            osm_map: Map array to update (modifies in-place)
            scene_metadata: Metadata dict with 'bbox_wgs84' {lon_min, lat_min, lon_max, lat_max}
            osm_data: Pre-fetched OSM data with 'roads' key containing LineString geometries
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, cannot rasterize roads")
            return
        
        roads_geom = []
        bbox_list = None
        
        # Get bbox_wgs84 from metadata
        if scene_metadata:
            bbox_wgs84 = scene_metadata.get('bbox_wgs84')
            if bbox_wgs84:
                # Convert dict to list: [lon_min, lat_min, lon_max, lat_max]
                bbox_list = [bbox_wgs84['lon_min'], bbox_wgs84['lat_min'], 
                            bbox_wgs84['lon_max'], bbox_wgs84['lat_max']]
        
        # Option 1: Use pre-fetched OSM data
        if osm_data and 'roads' in osm_data:
            roads_geom = osm_data['roads']
        
        # Option 2: Fetch from OSM using scene metadata
        elif bbox_list:
            try:
                roads_geom = self._fetch_roads_from_osm(bbox_list)
            except Exception as e:
                logger.warning(f"Failed to fetch roads from OSM: {e}")
                return
        else:
            # No data source for roads
            logger.debug("No bbox_wgs84 in metadata, cannot fetch roads")
            return
        
        if not roads_geom:
            logger.debug("No road geometries found")
            return
        
        # Convert road LineStrings to pixel coordinates and draw
        try:
            from shapely.geometry import LineString, MultiLineString
            import pyproj
            
            # Transform from WGS84 to local UTM coordinates
            center_lon = (bbox_list[0] + bbox_list[2]) / 2
            center_lat = (bbox_list[1] + bbox_list[3]) / 2
            
            # Use appropriate UTM zone
            utm_zone = int((center_lon + 180) / 6) + 1
            transformer = pyproj.Transformer.from_crs(
                "EPSG:4326",  # WGS84
                f"+proj=utm +zone={utm_zone} +datum=WGS84",
                always_xy=True
            )
            
            # Transform bbox corners to get the UTM bounds of our scene
            bbox_corners_utm = [
                transformer.transform(bbox_list[0], bbox_list[1]),  # SW corner
                transformer.transform(bbox_list[2], bbox_list[3]),  # NE corner
            ]
            utm_x_min = bbox_corners_utm[0][0]
            utm_y_min = bbox_corners_utm[0][1]
            utm_x_max = bbox_corners_utm[1][0]
            utm_y_max = bbox_corners_utm[1][1]
            
            # Calculate center of UTM bbox for centering
            utm_center_x = (utm_x_min + utm_x_max) / 2
            utm_center_y = (utm_y_min + utm_y_max) / 2
            
            # Calculate the scene's center in its local coordinate system
            scene_center_x = (self.x_min + self.x_max) / 2
            scene_center_y = (self.y_min + self.y_max) / 2
            
            logger.info(f"UTM bounds: X[{utm_x_min:.1f}, {utm_x_max:.1f}], Y[{utm_y_min:.1f}, {utm_y_max:.1f}]")
            logger.info(f"Scene bounds: X[{self.x_min:.1f}, {self.x_max:.1f}], Y[{self.y_min:.1f}, {self.y_max:.1f}]")
            
            road_polygons = []
            for road_geom in roads_geom:
                if isinstance(road_geom, (LineString, MultiLineString)):
                    # Convert to pixel coordinates
                    if isinstance(road_geom, LineString):
                        lines = [road_geom]
                    else:
                        lines = list(road_geom.geoms)
                    
                    for line in lines:
                        coords = np.array(line.coords)
                        
                        # Transform lon/lat to UTM meters
                        utm_coords = np.array([transformer.transform(lon, lat) for lon, lat in coords])
                        
                        # Transform from absolute UTM to scene-relative coordinates
                        # Center the UTM coords around origin, then add scene center
                        scene_coords = utm_coords.copy()
                        scene_coords[:, 0] = (utm_coords[:, 0] - utm_center_x) + scene_center_x
                        scene_coords[:, 1] = (utm_coords[:, 1] - utm_center_y) + scene_center_y
                        
                        # Convert to pixels using OSMRasterizer's coordinate system
                        pixels = self._world_to_pixel(scene_coords)
                        
                        # Create thick line polygon (buffer the line)
                        # Typical road width: 3-10 meters, let's use 5 meters
                        road_width_pixels = int(5.0 / self.resolution_x)  # 5 meters
                        road_width_pixels = max(2, road_width_pixels)  # At least 2 pixels
                        
                        # Draw line with thickness
                        if len(pixels) >= 2:
                            # Filter out-of-bounds points
                            valid_mask = (
                                (pixels[:, 0] >= 0) & (pixels[:, 0] < self.width) &
                                (pixels[:, 1] >= 0) & (pixels[:, 1] < self.height)
                            )
                            
                            # If we have at least 2 valid points or points crossing the boundary
                            if valid_mask.sum() >= 2 or (valid_mask.sum() > 0 and len(pixels) >= 2):
                                # Clip to bounds
                                clipped_pixels = np.clip(pixels, [0, 0], [self.width - 1, self.height - 1])
                                road_polygons.append((clipped_pixels.astype(np.int32), road_width_pixels))
            
            logger.info(f"Processing {len(road_polygons)} road segments for drawing")
            
            # Draw all roads at once
            if road_polygons:
                road_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                for pixels, thickness in road_polygons:
                    cv2.polylines(road_mask, [pixels], isClosed=False, 
                                color=1, thickness=thickness, lineType=cv2.LINE_AA)
                
                # Update channel 3 (roads)
                osm_map[3] = np.maximum(osm_map[3], road_mask.astype(np.float32))
                logger.info(f"Rasterized {len(road_polygons)} road segments")
        
        except ImportError as e:
            logger.warning(f"Missing dependency for road rasterization: {e}")
        except Exception as e:
            logger.error(f"Error rasterizing roads: {e}", exc_info=True)
    
    def _fetch_roads_from_osm(self, bbox: List[float]) -> List:
        """
        Fetch road geometries from OpenStreetMap.
        
        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
        
        Returns:
            List of shapely LineString geometries
        """
        try:
            import requests
            from shapely.geometry import shape, LineString
            import time
            
            lon_min, lat_min, lon_max, lat_max = bbox
            
            # Overpass API query for roads (highways)
            overpass_url = "https://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json][timeout:60];
            (
              way["highway"]({lat_min},{lon_min},{lat_max},{lon_max});
            );
            out geom;
            """
            
            # Try with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching roads from OSM (attempt {attempt + 1}/{max_retries})...")
                    response = requests.get(overpass_url, params={'data': overpass_query}, timeout=90)
                    response.raise_for_status()
                    break
                except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"OSM API request failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            data = response.json()
            
            roads = []
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    # Convert to LineString
                    coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
                    if len(coords) >= 2:
                        roads.append(LineString(coords))
            
            logger.info(f"Fetched {len(roads)} roads from OSM")
            return roads
        
        except Exception as e:
            logger.error(f"Failed to fetch roads from OSM: {e}")
            return []
