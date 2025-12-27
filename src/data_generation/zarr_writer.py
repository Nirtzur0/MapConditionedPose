"""
Zarr Dataset Writer for Multi-Layer Features
Efficient hierarchical storage with chunking and compression
"""

import numpy as np
import zarr
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

# Handle zarr v3 API changes
try:
    from numcodecs import Blosc, GZip, LZ4
    NUMCODECS_AVAILABLE = True
except ImportError:
    NUMCODECS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ZarrDatasetWriter:
    """
    Writes multi-layer features to Zarr dataset with efficient chunking.
    
    Dataset schema:
        rt_layer/
            path_gains: [N, max_paths] complex64
            path_delays: [N, max_paths] float32
            path_aoa_azimuth: [N, max_paths] float32
            path_aoa_elevation: [N, max_paths] float32
            path_aod_azimuth: [N, max_paths] float32
            path_aod_elevation: [N, max_paths] float32
            path_doppler: [N, max_paths] float32
            rms_delay_spread: [N] float32
            k_factor: [N] float32
            num_paths: [N] int32
        
        phy_fapi_layer/
            rsrp: [N, num_cells] float32
            rsrq: [N, num_cells] float32
            sinr: [N, num_cells] float32
            cqi: [N, num_cells] int32
            ri: [N, num_cells] int32
            pmi: [N, num_cells] int32
            l1_rsrp_beams: [N, num_beams] float32 (optional)
            best_beam_ids: [N, K] int32 (optional)
        
        mac_rrc_layer/
            serving_cell_id: [N] int32
            neighbor_cell_ids: [N, K] int32
            timing_advance: [N] int32
            dl_throughput_mbps: [N] float32
            bler: [N] float32
        
        positions/
            ue_x: [N] float32
            ue_y: [N] float32
            ue_z: [N] float32
        
        timestamps/
            t: [N] float32
        
        metadata/
            scene_ids: [N] object (string)
            scene_indices: [N] int16 (index into maps)
            ue_ids: [N] int32
        
        maps/
            radio_maps: [num_scenes, C, H, W] float32
            osm_maps: [num_scenes, C, H, W] float32
    """
    
    def __init__(self, 
                 output_dir: Path,
                 chunk_size: int = 100,
                 compression: str = 'blosc',
                 compression_level: int = 5):
        """
        Args:
            output_dir: Output directory for Zarr store
            chunk_size: Number of samples per chunk (default 100)
            compression: Compression algorithm ('blosc', 'gzip', 'lz4', None)
            compression_level: Compression level (1-9)
        """
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Zarr store
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.store_path = self.output_dir / f"dataset_{timestamp}.zarr"
        self.store = zarr.open(str(self.store_path), mode='w')
        
        # Track current index
        self.current_idx = 0
        
        # Track dataset statistics
        self.stats = {
            'num_samples': 0,
            'num_samples': 0,
            'num_scenes': 0,
            'scene_ids': [], # List of scene_id strings
            'scene_map_indices': {}, # Map scene_id -> index in maps array
        }
        
        logger.info(f"ZarrDatasetWriter initialized: {self.store_path}")
        logger.info(f"  Chunk size: {chunk_size}, Compression: {compression}")
    
    def append(self, scene_data: Dict[str, np.ndarray], scene_id: str, scene_metadata: Optional[Dict] = None):
        """
        Append data from one scene to the dataset.
        
        Args:
        Args:
            scene_data: Dictionary with all features and metadata
            scene_id: Scene identifier for tracking
            scene_metadata: Optional metadata for the scene (e.g., bbox)
            scene_idx: Optional explicit index for this scene (if linking to maps)
        """
        # Determine number of samples in this scene
        positions = scene_data.get('positions')
        if positions is None or len(positions) == 0:
            logger.warning(f"No data for scene {scene_id}, skipping")
            return
        
        num_samples = len(positions)
        logger.info(f"Appending {num_samples} samples from scene {scene_id}")
        
        # Track scene index
        if scene_id not in self.stats['scene_map_indices']:
            new_idx = len(self.stats['scene_map_indices'])
            self.stats['scene_map_indices'][scene_id] = new_idx
            # Add to list if not present (handled by set logic usually, but here strict)
            if scene_id not in self.stats['scene_ids']:
                 self.stats['scene_ids'].append(scene_id)
        
        current_scene_idx = self.stats['scene_map_indices'][scene_id]
        
        # Create arrays on first append
        if self.current_idx == 0:
            self._create_arrays(scene_data)
        
        # Write data
        end_idx = self.current_idx + num_samples
        
        # Resize all arrays to accommodate new data
        self._resize_arrays(end_idx)
        
        # Positions
        if 'positions' in scene_data:
            pos = scene_data['positions']
            self.store['positions/ue_x'][self.current_idx:end_idx] = pos[:, 0]
            self.store['positions/ue_y'][self.current_idx:end_idx] = pos[:, 1]
            self.store['positions/ue_z'][self.current_idx:end_idx] = pos[:, 2]
        
        # Timestamps
        if 'timestamps' in scene_data:
            self.store['timestamps/t'][self.current_idx:end_idx] = scene_data['timestamps']
        
        # RT layer
        self._write_layer(scene_data, 'rt', self.current_idx, end_idx)
        
        # PHY/FAPI layer
        self._write_layer(scene_data, 'phy_fapi', self.current_idx, end_idx)
        
        # MAC/RRC layer
        self._write_layer(scene_data, 'mac_rrc', self.current_idx, end_idx)
        
        # Metadata
        scene_ids = [scene_id] * num_samples
        self.store['metadata/scene_ids'][self.current_idx:end_idx] = scene_ids
        
        if scene_metadata and 'bbox' in scene_metadata:
            bbox = scene_metadata['bbox']
            bbox_array = np.array([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']], dtype=np.float32)
            self.store['metadata/scene_bbox'][self.current_idx:end_idx] = np.tile(bbox_array, (num_samples, 1))
        else:
            self.store['metadata/scene_bbox'][self.current_idx:end_idx] = np.zeros((num_samples, 4), dtype=np.float32)

        # Scene Indices
        self.store['metadata/scene_indices'][self.current_idx:end_idx] = np.full(num_samples, current_scene_idx, dtype=np.int16)

        
        # Update index and stats
        self.current_idx = end_idx
        self.stats['num_samples'] = end_idx
        self.stats['num_samples'] = end_idx
        # self.stats['num_scenes'] updated via write_scene_maps or implicitly tracked
        self.stats['num_scenes'] = len(self.stats['scene_ids'])
        
        logger.info(f"  Total samples: {self.current_idx}")
    
    def _create_arrays(self, sample_data: Dict[str, np.ndarray]):
        """
        Create Zarr arrays based on first sample.
        
        Args:
            sample_data: First scene data to infer shapes and dtypes
        """
        logger.info("Creating Zarr arrays...")
        
        # Positions (3D)
        self._create_array('positions/ue_x', dtype='float32', shape=(0,))
        self._create_array('positions/ue_y', dtype='float32', shape=(0,))
        self._create_array('positions/ue_z', dtype='float32', shape=(0,))
        
        # Timestamps
        self._create_array('timestamps/t', dtype='float32', shape=(0,))
        
        # RT layer
        self._create_layer_arrays('rt', sample_data)
        
        # PHY/FAPI layer
        self._create_layer_arrays('phy_fapi', sample_data)
        
        # MAC/RRC layer
        self._create_layer_arrays('mac_rrc', sample_data)
        
        # Metadata
        self._create_array('metadata/scene_ids', dtype='<U64', shape=(0,))  # Unicode string 64 chars
        self._create_array('metadata/scene_bbox', dtype='float32', shape=(0, 4))
        self._create_array('metadata/ue_ids', dtype='int32', shape=(0,))
        self._create_array('metadata/scene_indices', dtype='int16', shape=(0,))
        
        # Maps (growable by scene)
        if 'radio_maps' not in self.store:
            self._create_array('radio_maps', dtype='float32', shape=(0, 5, 256, 256)) 
        
        if 'osm_maps' not in self.store:
            self._create_array('osm_maps', dtype='float32', shape=(0, 5, 256, 256))
        
        # We store maps at root level for now as per previous schema expectation in dataset.py, 
        # or we update dataset.py to look in maps/. Let's stick to root based on dataset.py code:
        # self.store['radio_maps'][idx] -> Wait, dataset.py needs an update anyway.
        # Let's put them in root to match potential earlier expectations or `maps/` if preferred.
        # Reader expects: self.store['radio_maps'][idx] currently. 
        # But wait, the reader was failing because they didn't exist.
        # I will create them at root `radio_maps` and `osm_maps` to be simple.
        
        num_arrays = len([k for k in self.store.keys()])
        logger.info(f"Created {num_arrays} Zarr arrays")
    
    def _create_layer_arrays(self, layer_prefix: str, sample_data: Dict[str, np.ndarray]):
        """Create arrays for a specific layer."""
        for key, value in sample_data.items():
            if not key.startswith(f'{layer_prefix}/'):
                continue
            
            if isinstance(value, list):
                # Variable length - skip for now (advanced feature)
                logger.warning(f"Skipping variable length array: {key}")
                continue
            
            if value is None or (isinstance(value, np.ndarray) and value.size == 0):
                continue
            
            # Debug: check dtype
            logger.info(f"Creating array {key}: shape={value.shape}, dtype={value.dtype}")
            
            # Infer shape and dtype
            if value.ndim == 1:
                shape = (0,)
            elif value.ndim == 2:
                shape = (0, value.shape[1])
            elif value.ndim == 3:
                shape = (0, value.shape[1], value.shape[2])
            else:
                shape = tuple([0] + list(value.shape[1:]))
            
            dtype = value.dtype
            if dtype == np.complex128:
                dtype = np.complex64  # Reduce precision
            
            self._create_array(key, dtype=dtype, shape=shape)
    
    def _resize_arrays(self, new_size: int):
        """Resize all arrays to accommodate more data."""
        def resize_group(group):
            for key in group.keys():
                if key in ['radio_maps', 'osm_maps']:
                    # Do not resize scene-level maps with sample count
                    continue

                item = group[key]
                if isinstance(item, zarr.core.Array):
                    # Resize first dimension
                    current_shape = list(item.shape)
                    if current_shape[0] < new_size:
                        current_shape[0] = new_size
                        item.resize(tuple(current_shape))
                elif hasattr(item, 'keys'):
                    # Recursively resize nested groups
                    resize_group(item)
        
        resize_group(self.store)
    
    def _create_array(self, 
                     name: str,
                     dtype: Any,
                     shape: tuple,
                     fill_value: Optional[Any] = None):
        """
        Create a single Zarr array with proper chunking and compression.
        
        Args:
            name: Array path (with slashes for hierarchy)
            dtype: NumPy dtype
            shape: Initial shape (first dim should be 0 for appendable)
            fill_value: Fill value for uninitialized elements
        """
        # Infer chunk shape
        if len(shape) == 1:
            chunks = (self.chunk_size,)
        elif len(shape) == 2:
            chunks = (self.chunk_size, shape[1])
        elif len(shape) == 3:
            chunks = (self.chunk_size, shape[1], shape[2])
        else:
            chunks = tuple([self.chunk_size] + list(shape[1:]))
        
        # Get compressor for zarr v2
        compressor = None
        if self.compression and NUMCODECS_AVAILABLE:
            if self.compression == 'blosc':
                compressor = Blosc(cname='lz4', clevel=self.compression_level, shuffle=Blosc.BITSHUFFLE)
            elif self.compression == 'gzip':
                compressor = GZip(level=self.compression_level)
            elif self.compression == 'lz4':
                compressor = LZ4(acceleration=1)
        
        # Create resizable array with zarr v2 API
        # Set maxshape to None for unlimited growth along first dimension
        if len(shape) == 1:
            maxshape = (None,)
        elif len(shape) == 2:
            maxshape = (None, shape[1])
        elif len(shape) == 3:
            maxshape = (None, shape[1], shape[2])
        else:
            maxshape = tuple([None] + list(shape[1:]))
        
        arr = self.store.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value if fill_value is not None else 0,
            compressor=compressor,
            overwrite=True,
        )
        
        logger.debug(f"Created array {name}: shape={shape}, dtype={dtype}, chunks={chunks}")
        
        logger.debug(f"Created array {name}: shape={shape}, dtype={dtype}, chunks={chunks}")
    
    def _write_layer(self, 
                    scene_data: Dict[str, np.ndarray],
                    layer_prefix: str,
                    start_idx: int,
                    end_idx: int):
        """Write all arrays for a specific layer."""
        for key, value in scene_data.items():
            if not key.startswith(f'{layer_prefix}/'):
                continue
            
            if isinstance(value, list) or value is None:
                continue
            
            if value.size == 0:
                continue
            
            # Write to Zarr
            if key in self.store:
                # Resize if needed
                current_shape = self.store[key].shape
                required_shape = list(current_shape)
                required_shape[0] = end_idx
                
                if current_shape[0] < end_idx:
                    self.store[key].resize(required_shape)
                
                # Write data
                try:
                    self.store[key][start_idx:end_idx] = value
                except Exception as e:
                    logger.error(f"Error writing {key}: {e}")
                    logger.debug(f"  Shape: {value.shape}, Target: [{start_idx}:{end_idx}]")

    def write_scene_maps(self, scene_id: str, radio_map: np.ndarray, osm_map: np.ndarray):
        """
        Write radio and OSM maps for a specific scene.
        
        Args:
            scene_id: Scene ID
            radio_map: [C, H, W] array
            osm_map: [C, H, W] array
        """
        if scene_id not in self.stats['scene_map_indices']:
            new_idx = len(self.stats['scene_map_indices'])
            self.stats['scene_map_indices'][scene_id] = new_idx
            if scene_id not in self.stats['scene_ids']:
                 self.stats['scene_ids'].append(scene_id)
        
        idx = self.stats['scene_map_indices'][scene_id]
        
        # Ensure arrays exist (if created before first append)
        if 'radio_maps' not in self.store:
             # Infer shape from input
             C, H, W = radio_map.shape
             self._create_array('radio_maps', dtype='float32', shape=(0, C, H, W))
             self._create_array('osm_maps', dtype='float32', shape=(0, osm_map.shape[0], H, W))
             # Also ensure metadata arrays created if not yet
             if 'metadata/scene_indices' not in self.store:
                 self._create_array('metadata/scene_indices', dtype='int16', shape=(0,))


        # Resize if needed
        current_size = self.store['radio_maps'].shape[0]
        if idx >= current_size:
            new_size = idx + 1
            self.store['radio_maps'].resize(new_size, *self.store['radio_maps'].shape[1:])
            self.store['osm_maps'].resize(new_size, *self.store['osm_maps'].shape[1:])
        
        # Write
        self.store['radio_maps'][idx] = radio_map
        self.store['osm_maps'][idx] = osm_map
        logger.info(f"Wrote maps for scene {scene_id} at index {idx}")
    
    def finalize(self) -> Path:
        """
        Finalize dataset and write metadata.
        
        Returns:
            store_path: Path to completed Zarr store
        """
        # Write dataset-level metadata
        metadata = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'num_samples': self.stats['num_samples'],
            'num_scenes': self.stats['num_scenes'],
            'scene_ids': self.stats['scene_ids'],
            'chunk_size': self.chunk_size,
            'compression': self.compression,
            'coordinate_system': 'local_bottom_left',  # Explicitly define stored system
        }
        
        # Write to .zattrs
        self.store.attrs.update(metadata)
        
        # Write human-readable metadata.json
        metadata_path = self.store_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset finalized: {self.store_path}")
        logger.info(f"  Total samples: {self.stats['num_samples']}")
        logger.info(f"  Total scenes: {self.stats['num_scenes']}")
        
        # Compute size on disk
        total_size = sum(f.stat().st_size for f in self.store_path.rglob('*') if f.is_file())
        size_mb = total_size / 1024**2
        logger.info(f"  Size on disk: {size_mb:.1f} MB")
        
        return self.store_path
    
    def get_stats(self) -> Dict:
        """Get current dataset statistics."""
        return self.stats.copy()


def load_zarr_dataset(store_path: Path) -> zarr.Group:
    """
    Load a Zarr dataset for reading.
    
    Args:
        store_path: Path to .zarr directory
        
    Returns:
        store: Zarr group with hierarchical arrays
    """
    store = zarr.open(str(store_path), mode='r')
    logger.info(f"Loaded Zarr dataset: {store_path}")
    num_samples = store.attrs.get('num_samples', 'unknown')
    num_scenes = store.attrs.get('num_scenes', 'unknown')
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Scenes: {num_scenes}")
    
    return store


def zarr_to_dict(store: zarr.Group, 
                indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Convert Zarr store to dictionary (for PyTorch DataLoader).
    
    Args:
        store: Zarr group
        indices: Sample indices to load (if None, load all)
        
    Returns:
        data_dict: Dictionary with all arrays
    """
    if indices is None:
        num_samples = store.attrs.get('num_samples', len(store['positions/ue_x']))
        indices = np.arange(num_samples)
    
    data = {}
    
    # Recursively load all arrays
    def load_group(group, prefix=''):
        for key in group.keys():
            full_key = f"{prefix}{key}" if prefix else key
            item = group[key]
            
            if isinstance(item, zarr.Group):
                load_group(item, prefix=f"{full_key}/")
            else:
                # Load array at indices
                data[full_key] = item[indices]
    
    load_group(store)
    
    return data


if __name__ == "__main__":
    # Test Zarr writer
    logger.info("Testing ZarrDatasetWriter...")
    
    # Create test data
    num_samples = 250
    num_cells = 3
    max_paths = 50
    
    scene_data_1 = {
        'positions': np.random.uniform(-500, 500, (num_samples, 3)),
        'timestamps': np.arange(num_samples) * 0.2,
        'rt/path_gains': np.random.randn(num_samples, max_paths) + 1j * np.random.randn(num_samples, max_paths),
        'rt/path_delays': np.sort(np.random.uniform(0, 1e-6, (num_samples, max_paths)), axis=-1),
        'rt/rms_delay_spread': np.random.uniform(10e-9, 200e-9, num_samples),
        'phy_fapi/rsrp': np.random.uniform(-100, -60, (num_samples, num_cells)),
        'phy_fapi/rsrq': np.random.uniform(-15, -5, (num_samples, num_cells)),
        'phy_fapi/cqi': np.random.randint(0, 16, (num_samples, num_cells)),
        'mac_rrc/serving_cell_id': np.random.randint(0, num_cells, num_samples),
        'mac_rrc/timing_advance': np.random.randint(0, 1000, num_samples),
    }
    
    # Initialize writer
    output_dir = Path("test_zarr_output")
    output_dir.mkdir(exist_ok=True)
    
    writer = ZarrDatasetWriter(output_dir=output_dir, chunk_size=100)
    
    # Append scene 1
    writer.append(scene_data_1, scene_id='scene_001')
    logger.info(f"✓ Appended scene_001: {num_samples} samples")
    
    # Create and append scene 2
    scene_data_2 = scene_data_1.copy()
    scene_data_2['positions'] = np.random.uniform(-500, 500, (150, 3))
    scene_data_2['timestamps'] = np.arange(150) * 0.2
    for key in scene_data_2:
        if key.startswith(('rt/', 'phy_fapi/', 'mac_rrc/')) and isinstance(scene_data_2[key], np.ndarray):
            # Resize to 150 samples
            if scene_data_2[key].ndim == 1:
                scene_data_2[key] = scene_data_2[key][:150]
            else:
                scene_data_2[key] = scene_data_2[key][:150]
    
    writer.append(scene_data_2, scene_id='scene_002')
    logger.info(f"✓ Appended scene_002: 150 samples")
    
    # Finalize
    store_path = writer.finalize()
    logger.info(f"✓ Finalized dataset: {store_path}")
    
    # Test loading
    store = load_zarr_dataset(store_path)
    logger.info(f"✓ Loaded dataset: {store.tree()}")
    
    # Test conversion to dict
    data_dict = zarr_to_dict(store, indices=np.arange(10))
    logger.info(f"✓ Converted to dict: {len(data_dict)} arrays")
    logger.info(f"  - positions/ue_x: {data_dict['positions/ue_x'].shape}")
    logger.info(f"  - phy_fapi/rsrp: {data_dict['phy_fapi/rsrp'].shape}")
    
    logger.info("\nZarrDatasetWriter tests passed! ✓")
