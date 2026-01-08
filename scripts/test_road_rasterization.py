#!/usr/bin/env python3
"""
Test script to verify OSM road rasterization works.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data_generation.osm_rasterizer import OSMRasterizer


def test_road_fetching():
    """Test fetching roads from OSM."""
    print("Testing OSM road fetching...")
    
    # Example: Downtown San Francisco bbox
    bbox_wgs84 = {
        'lon_min': -122.4194,
        'lat_min': 37.7749,
        'lon_max': -122.4094,
        'lat_max': 37.7849
    }
    
    # Create rasterizer (approximate bounds in meters)
    # SF is roughly at UTM zone 10N
    # This is a rough approximation for testing
    x_min, y_min = -500, -500
    x_max, y_max = 500, 500
    
    rasterizer = OSMRasterizer(
        width=256,
        height=256,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
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
    
    # Test road rasterization
    print("Fetching and rasterizing roads...")
    rasterizer._rasterize_roads(osm_map, scene_metadata=metadata)
    
    # Check if roads were added
    road_channel = osm_map[3]
    if road_channel.sum() > 0:
        print(f"✓ Roads successfully rasterized!")
        print(f"  Road pixels: {(road_channel > 0).sum()}")
        print(f"  Coverage: {(road_channel > 0).sum() / road_channel.size * 100:.2f}%")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(road_channel, cmap='binary', origin='lower')
        axes[0].set_title('Road Channel (Binary)')
        axes[0].axis('off')
        
        axes[1].imshow(road_channel, cmap='hot', origin='lower')
        axes[1].set_title('Road Channel (Heatmap)')
        axes[1].axis('off')
        
        output_path = Path(__file__).parent.parent / 'outputs' / 'test_roads.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        
        return True
    else:
        print("✗ No roads found in channel 3")
        return False


if __name__ == '__main__':
    try:
        success = test_road_fetching()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
