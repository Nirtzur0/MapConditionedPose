"""
Pytest Test Suite for M1 Scene Generation
Validates deep integration with Geo2SigMap
"""

import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scene_generation import (
    SceneGenerator,
    MaterialRandomizer,
    SitePlacer,
    TileGenerator,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def test_bounds():
    """Standard test bounds (500m x 500m)."""
    return (0, 0, 500, 500)


@pytest.fixture
def material_randomizer(seed):
    """MaterialRandomizer with fixed seed."""
    return MaterialRandomizer(enable_randomization=True, seed=seed)


@pytest.fixture
def material_randomizer_deterministic():
    """MaterialRandomizer with randomization disabled."""
    return MaterialRandomizer(enable_randomization=False)


@pytest.fixture
def site_placer_grid(seed):
    """SitePlacer with grid strategy."""
    return SitePlacer(strategy="grid", seed=seed)


@pytest.fixture
def site_placer_random(seed):
    """SitePlacer with random strategy."""
    return SitePlacer(strategy="random", seed=seed)


@pytest.fixture
def site_placer_isd(seed):
    """SitePlacer with ISD strategy."""
    return SitePlacer(strategy="isd", seed=seed)


@pytest.fixture
def tile_generator(seed):
    """TileGenerator instance."""
    material_randomizer = MaterialRandomizer(seed=seed)
    site_placer = SitePlacer(strategy="grid", seed=seed)
    return TileGenerator(
        material_randomizer=material_randomizer,
        site_placer=site_placer,
    )


@pytest.fixture
def geo2sigmap_path():
    """Path to geo2sigmap package."""
    return "/home/ubuntu/projects/geo2sigmap/package/src"


@pytest.fixture
def boulder_bbox():
    """Boulder, CO test bounding box."""
    return (-105.30, 40.00, -105.25, 40.03)


# ============================================================================
# Material Randomizer Tests
# ============================================================================

class TestMaterialRandomizer:
    """Test suite for MaterialRandomizer."""
    
    def test_sample_reproducible_and_valid(self, seed):
        """Test sampling reproducibility and ITU material IDs."""
        rand1 = MaterialRandomizer(enable_randomization=True, seed=seed)
        rand2 = MaterialRandomizer(enable_randomization=True, seed=seed)

        mat1 = rand1.sample()
        mat2 = rand2.sample()

        assert mat1 == mat2
        assert set(mat1.keys()) == {'ground', 'rooftop', 'wall'}
        assert all(mat.startswith('mat-itu_') for mat in mat1.values())
    
    def test_sample_deterministic_mode(self, material_randomizer_deterministic):
        """Test deterministic mode always returns same materials."""
        mat1 = material_randomizer_deterministic.sample()
        mat2 = material_randomizer_deterministic.sample()
        
        assert mat1 == mat2
        assert mat1 == {
            'ground': 'mat-itu_wet_ground',
            'rooftop': 'mat-itu_metal',
            'wall': 'mat-itu_concrete',
        }
    
    def test_material_properties(self, material_randomizer):
        """Test material property retrieval."""
        props = material_randomizer.sample_properties('concrete')
        
        assert 'epsilon_r' in props
        assert 'sigma' in props
        assert 'freq_range_ghz' in props
        assert 'category' in props
        
        # Validate ranges
        assert 5.0 <= props['epsilon_r'] <= 8.0
        assert 0.01 <= props['sigma'] <= 0.1
        assert props['category'] == 'building'
    
    def test_get_material_properties_unknown_id(self, material_randomizer):
        """Test unknown ITU ID falls back to defaults."""
        props = material_randomizer.get_material_properties('mat-itu_unknown')

        assert props['epsilon_r'] > 0
        assert props['sigma'] > 0
        assert props['freq_range_ghz'][0] > 0
    
    def test_list_materials_by_category(self, material_randomizer):
        """Test filtering materials by category."""
        ground_mats = material_randomizer.list_materials(category='ground')
        building_mats = material_randomizer.list_materials(category='building')
        rooftop_mats = material_randomizer.list_materials(category='rooftop')
        
        assert len(ground_mats) >= 3  # wet, medium_dry, very_dry
        assert len(building_mats) >= 4  # concrete, brick, wood, glass
        assert len(rooftop_mats) >= 1  # metal
        
        # Check categories
        assert all(cfg['category'] == 'ground' for cfg in ground_mats.values())
        assert all(cfg['category'] == 'building' for cfg in building_mats.values())
        assert all(cfg['category'] == 'rooftop' for cfg in rooftop_mats.values())
    
    def test_invalid_material_name(self, material_randomizer):
        """Test error handling for invalid material."""
        with pytest.raises(ValueError):
            material_randomizer.sample_properties('invalid_material')


# ============================================================================
# Site Placer Tests
# ============================================================================

class TestSitePlacer:
    """Test suite for SitePlacer."""
    
    def test_init(self):
        """Test SitePlacer initialization."""
        placer = SitePlacer(strategy="grid")
        assert placer.strategy == "grid"
        assert placer.default_antenna is not None
    
    def test_grid_placement(self, site_placer_grid, test_bounds):
        """Test grid placement strategy."""
        sites = site_placer_grid.place(test_bounds, num_tx=2, num_rx=5)
        
        # Check site counts (TX sites have 3 sectors each)
        tx_sites = [s for s in sites if s.site_type == 'tx']
        rx_sites = [s for s in sites if s.site_type == 'rx']
        
        assert len(tx_sites) == 2 * 3  # 2 TX x 3 sectors
        assert len(rx_sites) == 5
        
        # Check TX sites have correct sectors
        for i in range(2):
            sectors = [s for s in tx_sites if s.cell_id == i]
            assert len(sectors) == 3
            azimuths = [s.antenna.orientation[0] for s in sectors]
            assert set(azimuths) == {0.0, 120.0, 240.0}
    
    def test_random_placement(self, site_placer_random, test_bounds):
        """Test random placement strategy."""
        sites = site_placer_random.place(test_bounds, num_tx=3, num_rx=10)
        
        tx_sites = [s for s in sites if s.site_type == 'tx']
        rx_sites = [s for s in sites if s.site_type == 'rx']
        
        assert len(tx_sites) == 3
        assert len(rx_sites) == 10
        
        # Check positions are within bounds (with margins)
        xmin, ymin, xmax, ymax = test_bounds
        for site in sites:
            x, y, z = site.position
            assert xmin <= x <= xmax
            assert ymin <= y <= ymax
    
    def test_isd_placement(self, site_placer_isd, test_bounds):
        """Test ISD (hexagonal grid) placement strategy."""
        sites = site_placer_isd.place(test_bounds, num_rx=10, isd_meters=200)
        
        tx_sites = [s for s in sites if s.site_type == 'tx']
        rx_sites = [s for s in sites if s.site_type == 'rx']
        
        # Should have multiple TX sites in 500x500m area with 200m ISD
        assert len(tx_sites) >= 3  # At least 1 site x 3 sectors
        assert len(rx_sites) == 10
        
        # Check sectors for each cell
        cell_ids = set(s.cell_id for s in tx_sites)
        for cell_id in cell_ids:
            sectors = [s for s in tx_sites if s.cell_id == cell_id]
            assert len(sectors) == 3  # 3-sector site
    
    def test_custom_placement(self, seed):
        """Test custom placement strategy."""
        custom_positions = {
            'tx': [(100, 200, 25), (300, 400, 25)],
            'rx': [(50, 50, 1.5), (150, 150, 1.5), (250, 250, 1.5)],
        }
        
        placer = SitePlacer(strategy="custom", seed=seed)
        sites = placer.place((0, 0, 500, 500), custom_positions=custom_positions)
        
        tx_sites = [s for s in sites if s.site_type == 'tx']
        rx_sites = [s for s in sites if s.site_type == 'rx']
        
        assert len(tx_sites) == 2
        assert len(rx_sites) == 3
        
        # Check positions match
        tx_positions = [s.position for s in tx_sites]
        assert (100, 200, 25) in tx_positions
        assert (300, 400, 25) in tx_positions
    
    def test_site_structure(self, site_placer_grid, test_bounds):
        """Test Site object structure."""
        sites = site_placer_grid.place(test_bounds, num_tx=1, num_rx=1)
        
        for site in sites:
            assert hasattr(site, 'site_id')
            assert hasattr(site, 'position')
            assert hasattr(site, 'site_type')
            assert hasattr(site, 'antenna')
            assert site.site_type in ['tx', 'rx']
            assert len(site.position) == 3  # (x, y, z)
            
            # Check to_dict() method
            site_dict = site.to_dict()
            assert 'site_id' in site_dict
            assert 'position' in site_dict
            assert 'antenna' in site_dict
    
    def test_invalid_strategy(self, seed):
        """Test error handling for invalid strategy."""
        placer = SitePlacer(strategy="invalid", seed=seed)
        with pytest.raises(ValueError):
            placer.place((0, 0, 500, 500), num_tx=1, num_rx=1)


# ============================================================================
# Tile Generator Tests
# ============================================================================

class TestTileGenerator:
    """Test suite for TileGenerator."""
    
    def test_create_tile_grid_structure_and_coverage(self, tile_generator, boulder_bbox):
        """Test tile grid structure and coverage."""
        tiles = tile_generator._create_tile_grid(
            bbox_wgs84=boulder_bbox,
            tile_size_meters=500,
            overlap_meters=50,
        )
        
        assert len(tiles) > 0
        
        # Check tile structure
        for tile in tiles:
            assert 'tile_id' in tile
            assert 'tile_x' in tile
            assert 'tile_y' in tile
            assert 'bounds_utm' in tile
            assert 'polygon_wgs84' in tile
            assert 'utm_zone' in tile
            assert 'hemisphere' in tile
            
            # Check bounds format
            bounds = tile['bounds_utm']
            assert len(bounds) == 4  # (xmin, ymin, xmax, ymax)
            
            # Check polygon format
            polygon = tile['polygon_wgs84']
            assert len(polygon) == 5  # Closed polygon
            assert polygon[0] == polygon[-1]  # First == last
        
        # Check that tiles form a complete grid (no gaps)
        tile_xs = set(t['tile_x'] for t in tiles)
        tile_ys = set(t['tile_y'] for t in tiles)
        for x in range(max(tile_xs) + 1):
            for y in range(max(tile_ys) + 1):
                matching = [t for t in tiles if t['tile_x'] == x and t['tile_y'] == y]
                assert len(matching) == 1
    
    def test_utm_to_wgs84_polygon(self, tile_generator):
        """Test UTM to WGS84 conversion."""
        from pyproj import Transformer
        
        # Create transformer for UTM zone 13N (Boulder area)
        transformer = Transformer.from_crs(
            "EPSG:4326",
            "+proj=utm +zone=13 +north +datum=WGS84",
            always_xy=True
        )
        
        bounds_utm = (500000, 4428000, 500500, 4428500)
        polygon = tile_generator._utm_to_wgs84_polygon(bounds_utm, transformer)
        
        assert len(polygon) == 5
        assert polygon[0] == polygon[-1]
        
        # Check all points are valid WGS84
        for lon, lat in polygon:
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90
    
# ============================================================================
# Scene Generator Tests  
# ============================================================================

class TestSceneGenerator:
    """Test suite for SceneGenerator."""
    
    def test_init_without_geo2sigmap(self):
        """Test that geo2sigmap_path is ignored when module is installed."""
        pytest.importorskip("geo2sigmap")
        material_randomizer = MaterialRandomizer()
        site_placer = SitePlacer()

        scene_gen = SceneGenerator(
            geo2sigmap_path="/invalid/path",
            material_randomizer=material_randomizer,
            site_placer=site_placer,
        )

        assert scene_gen.scene is not None
    
    @pytest.mark.skipif(
        not Path("/home/ubuntu/projects/geo2sigmap/package/src").exists(),
        reason="Geo2SigMap not installed"
    )
    def test_init_with_geo2sigmap(self, geo2sigmap_path):
        """Test SceneGenerator initialization with valid geo2sigmap."""
        try:
            import shapely
        except ImportError:
            pytest.skip("Shapely not installed (required by geo2sigmap)")
        
        material_randomizer = MaterialRandomizer(seed=42)
        site_placer = SitePlacer(strategy="grid", seed=42)
        
        scene_gen = SceneGenerator(
            geo2sigmap_path=geo2sigmap_path,
            material_randomizer=material_randomizer,
            site_placer=site_placer,
        )
        
        assert scene_gen.material_randomizer is not None
        assert scene_gen.site_placer is not None
        assert scene_gen.scene is not None  # Geo2SigMap Scene loaded


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_m1_pipeline_structure(self, material_randomizer, site_placer_grid, test_bounds):
        """Test the complete M1 pipeline structure (without Geo2SigMap)."""
        # Step 1: Sample materials
        materials = material_randomizer.sample()
        assert 'ground' in materials
        assert 'wall' in materials
        assert 'rooftop' in materials
        
        # Step 2: Place sites
        sites = site_placer_grid.place(test_bounds, num_tx=2, num_rx=10)
        tx_sites = [s for s in sites if s.site_type == 'tx']
        rx_sites = [s for s in sites if s.site_type == 'rx']
        
        # Step 3: Create metadata structure (simulated)
        metadata = {
            'scene_id': 'test_scene_001',
            'bounds': test_bounds,
            'materials': materials,
            'sites': [s.to_dict() for s in sites],
            'num_tx': len(tx_sites),
            'num_rx': len(rx_sites),
        }
        
        # Verify metadata structure
        assert metadata['scene_id'] == 'test_scene_001'
        assert len(metadata['sites']) == len(sites)
        assert metadata['num_tx'] == len(tx_sites)
        assert metadata['num_rx'] == len(rx_sites)
        
        # Verify all site dicts are valid
        for site_dict in metadata['sites']:
            assert 'site_id' in site_dict
            assert 'position' in site_dict
            assert 'site_type' in site_dict
            assert 'antenna' in site_dict
