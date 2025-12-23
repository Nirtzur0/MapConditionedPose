# M1 Scene Generation Architecture

## Deep Integration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    transformer-ue-localization                      │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ SceneGenerator (core.py)                                   │   │
│  │                                                            │   │
│  │  ┌──────────────────────────────────────────────┐         │   │
│  │  │ Geo2SigMapScene (IMPORTED DIRECTLY)         │         │   │
│  │  │ ┌────────────────────────────────────────┐  │         │   │
│  │  │ │ • OSM building extraction             │  │         │   │
│  │  │ │ • Terrain meshing (DEM/LiDAR)        │  │         │   │
│  │  │ │ • Material assignment                 │  │         │   │
│  │  │ │ • Mitsuba XML generation              │  │         │   │
│  │  │ └────────────────────────────────────────┘  │         │   │
│  │  └──────────────────────────────────────────────┘         │   │
│  │                        ↓                                   │   │
│  │  ┌──────────────────────────────────────────────┐         │   │
│  │  │ OUR EXTENSIONS (not wrapping, extending)    │         │   │
│  │  │                                             │         │   │
│  │  │  MaterialRandomizer:                        │         │   │
│  │  │    • Sample ITU materials (ground/walls)    │         │   │
│  │  │    • Domain randomization for sim-to-real   │         │   │
│  │  │                                             │         │   │
│  │  │  SitePlacer:                                │         │   │
│  │  │    • Grid/random/ISD placement strategies   │         │   │
│  │  │    • 3-sector BS configuration              │         │   │
│  │  │    • Antenna patterns (3GPP 38.901)         │         │   │
│  │  │                                             │         │   │
│  │  │  XML Modification:                          │         │   │
│  │  │    • Add <sensor> tags for Sionna RT        │         │   │
│  │  │    • Transmitter/receiver positions         │         │   │
│  │  │    • Antenna orientations                   │         │   │
│  │  │                                             │         │   │
│  │  │  Metadata Generation:                       │         │   │
│  │  │    • Site positions + antenna configs       │         │   │
│  │  │    • Material properties for M2             │         │   │
│  │  │    • Tile coordinates (UTM/WGS84)           │         │   │
│  │  └──────────────────────────────────────────────┘         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ TileGenerator (tiles.py)                                   │   │
│  │  • Batch processing for large areas                        │   │
│  │  • WGS84 ↔ UTM coordinate transforms                       │   │
│  │  • Grid tiling with overlap                                │   │
│  │  • Metadata aggregation                                    │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT (M1 → M2)                            │
│                                                                     │
│  data/scenes/scene_001/                                             │
│  ├── scene.xml           ← Mitsuba scene (Geo2SigMap + our sites)  │
│  ├── buildings.obj       ← Building geometry (from Geo2SigMap)     │
│  ├── terrain.obj         ← Terrain mesh (from Geo2SigMap)          │
│  └── metadata.json       ← Extended metadata:                      │
│                             {                                       │
│                               "scene_id": "...",                    │
│                               "bounds": [...],                      │
│                               "materials": {                        │
│                                 "ground": "mat-itu_wet_ground",     │
│                                 "wall": "mat-itu_concrete",         │
│                                 "rooftop": "mat-itu_metal"          │
│                               },                                    │
│                               "sites": [                            │
│                                 {                                   │
│                                   "site_id": "tx_0_sector_0",       │
│                                   "position": [x, y, z],            │
│                                   "site_type": "tx",                │
│                                   "antenna": {                      │
│                                     "pattern": "3gpp_38901",        │
│                                     "orientation": [0, 10, 0],      │
│                                     "polarization": "dual"          │
│                                   },                                │
│                                   "cell_id": 0,                     │
│                                   "sector_id": 0,                   │
│                                   "power_dbm": 43.0                 │
│                                 },                                  │
│                                 ...                                 │
│                               ],                                    │
│                               "num_buildings": 42,                  │
│                               "tile": {...}                         │
│                             }                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Direct Integration (NOT Wrapping)

**❌ What we DIDN'T do:**
```python
class SceneWrapper:
    def __init__(self):
        self._geo2sigmap = Geo2SigMap()  # Hiding it
    
    def generate(self):
        # Wrapper API that hides geo2sigmap
        return self._geo2sigmap.do_something()
```

**✓ What we DID do:**
```python
class SceneGenerator:
    def __init__(self):
        # Import geo2sigmap Scene DIRECTLY
        self.scene = Geo2SigMapScene()
        # Add our extensions
        self.material_randomizer = MaterialRandomizer()
        self.site_placer = SitePlacer()
    
    def generate(self, ...):
        # Call geo2sigmap directly
        self.scene(points=polygon_points, ...)
        
        # Extend its output
        materials = self.material_randomizer.sample()
        sites = self.site_placer.place(...)
        self._add_sites_to_xml(sites)
        
        return metadata
```

### 2. Extension Points

We extend Geo2SigMap at 3 key points:

1. **Pre-generation**: Material sampling
   - Randomize ITU materials before calling Scene
   - Domain randomization for robust training

2. **Post-generation**: Site placement
   - Calculate TX/RX positions based on scene bounds
   - Generate antenna configurations
   - Modify Mitsuba XML to add sensors

3. **Metadata enrichment**: UE localization specifics
   - Site positions, antenna configs, cell IDs
   - Material properties (ε_r, σ) for channel modeling
   - Coordinate transforms (UTM zones)

### 3. Coordinate System Flow

```
User Input (WGS84)
    │ lon/lat polygon
    │
    ├─→ Geo2SigMap
    │   └─→ Overpass API query (WGS84)
    │   └─→ Blender scene generation (local coords)
    │   └─→ Mitsuba XML (local meters)
    │
    └─→ TileGenerator
        └─→ Convert to UTM for tiling
        └─→ Track zone/hemisphere for consistency
        └─→ Metadata includes both WGS84 + UTM
```

### 4. Material Handling

```
MaterialRandomizer.sample()
    │
    ├─→ Returns ITU material IDs
    │   └─→ "mat-itu_concrete", "mat-itu_wet_ground", etc.
    │
    └─→ Pass to Geo2SigMap Scene(...)
        └─→ Geo2SigMap uses its ITU library
            └─→ Assigns to buildings/ground in XML
            
MaterialRandomizer.get_material_properties()
    │
    └─→ Returns physical properties (ε_r, σ)
        └─→ Saved to metadata.json
            └─→ Used by M2 for Sionna RT validation
```

### 5. Site Placement Strategies

```
SitePlacer.place(bounds, strategy=...)
    │
    ├─→ "grid": Uniform coverage
    │   └─→ 3-sector sites with 120° azimuth separation
    │   └─→ Regular grid spacing
    │
    ├─→ "random": Diverse scenarios  
    │   └─→ Uniform random TX/RX placement
    │   └─→ Random orientations
    │
    ├─→ "isd": 3GPP-style hexagonal grid
    │   └─→ Inter-site distance (200m urban, 500m suburban)
    │   └─→ 3-sector per hex site
    │
    └─→ "custom": User-specified positions
        └─→ From measurement campaigns
        └─→ From config files
```

## Comparison: Wrapper vs Deep Integration

| Aspect | Wrapper (❌ Rejected) | Deep Integration (✓ Implemented) |
|--------|----------------------|-----------------------------------|
| **Import** | `from geo2sigmap import Geo2SigMap`<br>Hides it in private `_geo2sigmap` | `from geo2sigmap import Scene`<br>Use directly as `self.scene` |
| **Usage** | `wrapper.generate()` calls hidden geo2sigmap | `self.scene(...)` called directly |
| **Extension** | New abstraction layer | Direct XML modification, metadata enrichment |
| **Maintenance** | Must update wrapper when geo2sigmap changes | Minimal impact - we call same methods |
| **Clarity** | User doesn't know geo2sigmap is used | Clear separation: geo2sigmap + our extensions |
| **Flexibility** | Limited by wrapper API | Full access to geo2sigmap capabilities |

## M1 → M2 Data Flow

```
M1: SceneGenerator.generate()
    ↓
    scene.xml           ← Mitsuba scene with sensors
    buildings.obj       ← Building geometry
    terrain.obj         ← Terrain mesh  
    metadata.json       ← Sites, materials, bounds
    
M2: MultiLayerDataGenerator (TODO)
    │
    ├─→ Load scene.xml in Sionna RT
    │   └─→ Configure TX/RX from metadata.json
    │   └─→ Run ray tracing → CIR, paths, angles
    │
    ├─→ Compute channel features
    │   └─→ RT layer: CIR, delay spread, AoA/AoD
    │   └─→ PHY layer: SINR, SNR, MCS
    │   └─→ FAPI layer: CQI, RI, PMI
    │   └─→ MAC/RRC layer: TA, RSRP, cell ID
    │
    └─→ Save to Zarr
        └─→ Zarr arrays: (num_scenes, num_rx, feature_dim)
        └─→ Metadata: scene_id → file mapping
        
M3: PyTorch DataLoader (TODO)
    └─→ Load from Zarr → torch.Tensor
        └─→ Radio encoder: attention over measurements
        └─→ Map encoder: ViT over OSM + radio maps  
        └─→ Fusion → position regression
```

## Summary

M1 achieves **deep integration** by:
1. **Importing** Geo2SigMap Scene directly (no wrappers)
2. **Using** it as-is for proven OSM→Mitsuba pipeline  
3. **Extending** with material randomization, site placement, metadata
4. **Outputting** scenes ready for M2 Sionna RT processing

This approach:
- ✓ Respects the original architecture
- ✓ Leverages proven Geo2SigMap code
- ✓ Adds UE localization requirements cleanly
- ✓ Maintains clear separation of concerns
- ✓ Enables easy debugging (no hidden abstractions)
