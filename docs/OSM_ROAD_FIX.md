# OSM Road Rasterization Fix

## Problem

When generating OSM layers, channel 3 (roads) was always empty. The rasterizer was looking for scene objects with "road" or "street" in their names, but Sionna's `load_scene()` only loads buildings as 3D meshes, not roads.

## Root Cause

1. **Sionna Scene Limitation**: Sionna RT's scene loader only converts OSM buildings to 3D meshes for ray tracing
2. **Missing Road Data**: Roads exist in OpenStreetMap as 2D LineStrings with 'highway' tags, but aren't loaded into the scene
3. **Incorrect Assumption**: The rasterizer expected road objects in `scene.objects`, but they don't exist

## Solution

Implemented a separate pipeline to fetch and rasterize roads directly from OpenStreetMap:

### 1. **Added `_rasterize_roads()` method** (lines 635-736 in osm_rasterizer.py)
   - Fetches roads from OSM using Overpass API
   - Transforms WGS84 coordinates → UTM meters → pixel coordinates
   - Draws roads with appropriate width (~5 meters) on channel 3
   - Handles LineString and MultiLineString geometries

### 2. **Added `_fetch_roads_from_osm()` method** (lines 738-777)
   - Queries Overpass API for highway ways within bbox
   - Returns list of Shapely LineString geometries
   - Includes proper error handling and logging

### 3. **Updated `rasterize()` signature** (line 118)
   - Now accepts `scene_metadata` and `osm_data` parameters
   - Calls `_rasterize_roads()` after processing scene objects
   - Backward compatible with existing code

### 4. **Updated callers in multi_layer_generator.py**
   - Line 692: Single scene generation passes `scene_metadata`
   - Line 1012: Batch OSM map generation passes `metadata`
   - Enables road fetching using bbox_wgs84 from scene metadata

### 5. **Dependencies**
   - Added `requests = "^2.31.0"` to pyproject.toml for Overpass API
   - Existing dependencies (shapely, pyproj, cv2) already satisfied

## Technical Details

### Coordinate Transform Pipeline
```
WGS84 (lon, lat) 
  → Overpass API Query
  → Shapely LineStrings
  → UTM meters (pyproj transform)
  → Pixel coordinates (OSMRasterizer._world_to_pixel)
  → OpenCV polylines drawing
```

### Metadata Structure
```python
scene_metadata = {
    'bbox_wgs84': {
        'lon_min': float,
        'lat_min': float,
        'lon_max': float,
        'lat_max': float
    },
    'bbox': {  # Projected meter bounds
        'x_min': float,
        'y_min': float,
        'x_max': float,
        'y_max': float
    }
}
```

## Testing

Created `scripts/test_road_rasterization.py` to verify:
- OSM API querying works
- Roads are correctly rasterized to channel 3
- Coordinate transforms are accurate
- Visualization shows road coverage

Run test:
```bash
python scripts/test_road_rasterization.py
```

## Benefits

1. **Complete OSM Maps**: All 5 channels now populated correctly
2. **API-Based**: No dependency on Sionna scene loader for roads
3. **Flexible**: Can fetch roads for any bbox_wgs84
4. **Cached**: Future enhancement could cache road geometries in scene metadata
5. **Accurate**: Uses official OSM highway data with proper coordinate transforms

## Future Enhancements

1. **Cache road data** in scene metadata during generation
2. **Support other OSM features**: railways, waterways, landuse
3. **Road width by type**: Primary roads wider than residential
4. **Offline mode**: Pre-download OSM data for bbox regions
5. **Multi-resolution**: Generate roads at different detail levels

## Commit

- Commit: `5ad3763`
- Message: "Fix OSM road rasterization - fetch roads directly from OSM API"
- Files Changed: 3
  - `src/data_generation/osm_rasterizer.py` (+144 lines)
  - `src/data_generation/multi_layer_generator.py` (+4 lines)
  - `pyproject.toml` (+1 line)
  - `scripts/test_road_rasterization.py` (new file)
