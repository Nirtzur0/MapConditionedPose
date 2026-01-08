#!/usr/bin/env python3
"""
Test script for OSM road rasterization using mock data (no API calls).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
from shapely.geometry import LineString

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data_generation.osm_rasterizer import OSMRasterizer


def test_road_rasterization_with_mock_data():
    """Test road rasterization with pre-defined mock road data."""
    print("Testing OSM road rasterization with mock data...")
    
    # Boulder, CO area
    bbox_wgs84 = {
        'lon_min': -105.2820,
        'lat_min': 40.0150,
        'lon_max': -105.2770,
        'lat_max': 40.0180
    }
    
    # Create some mock road LineStrings (in lon/lat)
    # These represent a simple grid of roads
    mock_roads = [
        # Horizontal roads
        LineString([(-105.2820, 40.0160), (-105.2770, 40.0160)]),
        LineString([(-105.2820, 40.0165), (-105.2770, 40.0165)]),
        LineString([(-105.2820, 40.0170), (-105.2770, 40.0170)]),
        # Vertical roads
        LineString([(-105.2800, 40.0150), (-105.2800, 40.0180)]),
        LineString([(-105.2790, 40.0150), (-105.2790, 40.0180)]),
    ]
    
    # Create rasterizer
    x_min, y_min = -250, -150
    x_max, y_max = 250, 150
    
    rasterizer = OSMRasterizer(
        map_size=(256, 256),
        map_extent=(x_min, y_min, x_max, y_max)
    )
    
    # Create empty OSM map
    osm_map = np.zeros((5, 256, 256), dtype=np.float32)
    
    # Create metadata
    metadata = {
        'bbox_wgs84': bbox_wgs84,
        'bbox': {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }
    }
    
    # Create mock OSM data with roads
    osm_data = {
        'roads': mock_roads
    }
    
    # Test road rasterization with mock data
    print(f"Rasterizing {len(mock_roads)} mock roads...")
    rasterizer._rasterize_roads(osm_map, scene_metadata=metadata, osm_data=osm_data)
    
    # Check if roads were added
    road_channel = osm_map[3]
    if road_channel.sum() > 0:
        print(f"✓ Roads successfully rasterized!")
        print(f"  Road pixels: {(road_channel > 0).sum()}")
        print(f"  Coverage: {(road_channel > 0).sum() / road_channel.size * 100:.2f}%")
        print(f"  Min value: {road_channel.min()}, Max value: {road_channel.max()}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(road_channel, cmap='binary', origin='lower')
        axes[0].set_title('Road Channel (Binary)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('on')
        
        axes[1].imshow(road_channel, cmap='hot', origin='lower')
        axes[1].set_title('Road Channel (Heatmap)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('on')
        
        output_path = Path(__file__).parent.parent / 'outputs' / 'test_roads_mock.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        
        return True
    else:
        print("✗ No roads found in channel 3")
        print(f"  Road channel stats: min={road_channel.min()}, max={road_channel.max()}, sum={road_channel.sum()}")
        return False


if __name__ == '__main__':
    try:
        success = test_road_rasterization_with_mock_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
