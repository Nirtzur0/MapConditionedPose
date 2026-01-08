"""
Convert Zarr dataset to LMDB for efficient multiprocessing data loading.

LMDB provides:
- Perfect multiprocessing support (no async event loop issues)
- Fast random access (memory-mapped)
- Simple key-value storage

Usage:
    python scripts/convert_zarr_to_lmdb.py \
        --zarr-path data/processed/sionna_dataset/dataset_20260107_144332.zarr \
        --lmdb-path data/processed/sionna_dataset/dataset_20260107_144332.lmdb \
        --map-size 50
"""

import argparse
import zarr
import lmdb
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_zarr_to_lmdb(zarr_path: str, lmdb_path: str, map_size_gb: int = 50):
    """Convert Zarr dataset to LMDB format.
    
    Args:
        zarr_path: Path to input Zarr store
        lmdb_path: Path to output LMDB database
        map_size_gb: Maximum database size in GB (LMDB requires pre-allocation)
    """
    zarr_path = Path(zarr_path)
    lmdb_path = Path(lmdb_path)
    
    logger.info(f"Opening Zarr store: {zarr_path}")
    zarr_store = zarr.open(str(zarr_path), mode='r')
    
    # Get total number of samples
    n_samples = zarr_store['positions/ue_x'].shape[0]
    logger.info(f"Found {n_samples} samples in Zarr dataset")
    
    # Create LMDB environment
    logger.info(f"Creating LMDB database: {lmdb_path}")
    logger.info(f"Pre-allocating {map_size_gb}GB of space")
    lmdb_path.mkdir(parents=True, exist_ok=True)
    
    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_gb * 1024**3,  # Convert GB to bytes
        writemap=True,  # Faster writes
        map_async=True,  # Async flush
    )
    
    # Store metadata separately
    logger.info("Converting metadata...")
    metadata = extract_metadata(zarr_store)
    with env.begin(write=True) as txn:
        txn.put(b'__metadata__', pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL))
    
    # Convert samples
    logger.info(f"Converting {n_samples} samples...")
    batch_size = 1000  # Commit every N samples for performance
    
    with env.begin(write=True) as txn:
        for idx in tqdm(range(n_samples), desc="Converting samples"):
            sample = extract_sample(zarr_store, idx, metadata)
            
            # Serialize and store
            key = f'sample_{idx:06d}'.encode()
            value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(key, value)
            
            # Periodic commit for safety
            if (idx + 1) % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
    
    # Final sync
    env.sync()
    env.close()
    
    # Verify conversion
    logger.info("Verifying conversion...")
    verify_conversion(lmdb_path, n_samples)
    
    logger.info(f"✅ Conversion complete! LMDB database at: {lmdb_path}")
    logger.info(f"   Database size: {get_dir_size(lmdb_path)/(1024**3):.2f} GB")


def extract_metadata(zarr_store) -> dict:
    """Extract global metadata from Zarr store."""
    metadata = {
        'n_samples': zarr_store['positions/ue_x'].shape[0],
        'normalization': {},
        'radio_maps': None,
        'osm_maps': None,
    }
    
    # Extract normalization stats
    if 'metadata/normalization' in zarr_store:
        norm_group = zarr_store['metadata/normalization']
        
        # RT normalization
        rt_stats = {}
        if 'rt' in norm_group:
            for key in norm_group['rt'].keys():
                if key in norm_group['rt'] and 'mean' in norm_group['rt'][key]:
                    rt_stats[key] = {
                        'mean': float(norm_group['rt'][key]['mean'][0]),
                        'std': float(norm_group['rt'][key]['std'][0]),
                    }
        metadata['normalization']['rt'] = rt_stats
        
        # PHY normalization
        phy_stats = {}
        if 'phy_fapi' in norm_group:
            for key in norm_group['phy_fapi'].keys():
                if key in norm_group['phy_fapi'] and 'mean' in norm_group['phy_fapi'][key]:
                    phy_stats[key] = {
                        'mean': float(norm_group['phy_fapi'][key]['mean'][0]),
                        'std': float(norm_group['phy_fapi'][key]['std'][0]),
                    }
        metadata['normalization']['phy'] = phy_stats
        
        # MAC normalization
        mac_stats = {}
        if 'mac_rrc' in norm_group:
            for key in norm_group['mac_rrc'].keys():
                if key in norm_group['mac_rrc'] and 'mean' in norm_group['mac_rrc'][key]:
                    mac_stats[key] = {
                        'mean': float(norm_group['mac_rrc'][key]['mean'][0]),
                        'std': float(norm_group['mac_rrc'][key]['std'][0]),
                    }
        metadata['normalization']['mac'] = mac_stats
    
    # Store radio and OSM maps (shared across samples)
    if 'radio_maps' in zarr_store:
        metadata['radio_maps'] = np.array(zarr_store['radio_maps'][:])
        logger.info(f"  Loaded radio_maps: {metadata['radio_maps'].shape}")
    
    if 'metadata/osm_maps' in zarr_store:
        metadata['osm_maps'] = np.array(zarr_store['metadata/osm_maps'][:])
        logger.info(f"  Loaded osm_maps: {metadata['osm_maps'].shape}")
    
    return metadata


def extract_sample(zarr_store, idx: int, metadata: dict) -> dict:
    """Extract a single sample from Zarr store.
    
    Returns dictionary with all data needed for training:
    - measurements: RT, PHY, MAC features
    - radio_map: Scene radio map
    - osm_map: Scene OSM map
    - position: Ground truth position
    - metadata: Scene info
    """
    sample = {}
    
    # Positions (ground truth)
    sample['position'] = np.array([
        zarr_store['positions/ue_x'][idx],
        zarr_store['positions/ue_y'][idx],
    ], dtype=np.float32)
    
    # RT features (16 features per cell, up to 20 cells in schema but dataset has 16)
    rt_data = {}
    n_cells = 16  # Actual data has 16 cells
    rt_data['toa'] = zarr_store['rt/toa'][idx, :n_cells]
    rt_data['path_gains'] = zarr_store['rt/path_gains'][idx, :n_cells, :]
    rt_data['path_delays'] = zarr_store['rt/path_delays'][idx, :n_cells, :]
    rt_data['num_paths'] = zarr_store['rt/num_paths'][idx, :n_cells]
    rt_data['rms_delay_spread'] = zarr_store['rt/rms_delay_spread'][idx, :n_cells]
    rt_data['rms_angular_spread'] = zarr_store['rt/rms_angular_spread'][idx, :n_cells]
    sample['rt'] = {k: np.array(v, dtype=np.float32) for k, v in rt_data.items()}
    
    # PHY features (8 features per cell)
    phy_data = {}
    phy_data['rsrp'] = zarr_store['phy_fapi/rsrp'][idx, :n_cells, :]
    phy_data['rsrq'] = zarr_store['phy_fapi/rsrq'][idx, :n_cells, :]
    phy_data['sinr'] = zarr_store['phy_fapi/sinr'][idx, :n_cells, :]
    phy_data['cqi'] = zarr_store['phy_fapi/cqi'][idx, :n_cells, :]
    phy_data['ri'] = zarr_store['phy_fapi/ri'][idx, :n_cells, :]
    phy_data['pmi'] = zarr_store['phy_fapi/pmi'][idx, :n_cells, :]
    phy_data['l1_rsrp_beams'] = zarr_store['phy_fapi/l1_rsrp_beams'][idx, :n_cells, :]
    phy_data['best_beam_ids'] = zarr_store['phy_fapi/best_beam_ids'][idx, :n_cells, :]
    sample['phy'] = {k: np.array(v, dtype=np.float32) for k, v in phy_data.items()}
    
    # MAC features (6 features per cell)
    mac_data = {}
    mac_data['serving_cell_id'] = zarr_store['mac_rrc/serving_cell_id'][idx, :n_cells]
    mac_data['neighbor_cell_ids'] = zarr_store['mac_rrc/neighbor_cell_ids'][idx, :n_cells, :]
    mac_data['timing_advance'] = zarr_store['mac_rrc/timing_advance'][idx, :n_cells]
    mac_data['dl_throughput_mbps'] = zarr_store['mac_rrc/dl_throughput_mbps'][idx, :n_cells]
    sample['mac'] = {k: np.array(v, dtype=np.float32) for k, v in mac_data.items()}
    
    # Scene index for map lookup
    if 'metadata/scene_indices' in zarr_store:
        sample['scene_index'] = int(zarr_store['metadata/scene_indices'][idx])
    else:
        sample['scene_index'] = 0  # Default to first scene
    
    # Scene metadata
    if 'metadata/scene_ids' in zarr_store:
        sample['scene_id'] = str(zarr_store['metadata/scene_ids'][idx])
    
    if 'metadata/scene_extent' in zarr_store:
        sample['scene_extent'] = float(zarr_store['metadata/scene_extent'][idx, 0])
    
    if 'metadata/scene_bbox' in zarr_store:
        sample['scene_bbox'] = np.array(zarr_store['metadata/scene_bbox'][idx], dtype=np.float32)
    
    # Number of actual cells/paths
    if 'metadata/actual_num_cells' in zarr_store:
        sample['actual_num_cells'] = int(zarr_store['metadata/actual_num_cells'][idx])
    else:
        sample['actual_num_cells'] = n_cells
    
    return sample


def verify_conversion(lmdb_path: Path, expected_samples: int):
    """Verify LMDB database integrity."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    
    with env.begin() as txn:
        # Check metadata
        metadata_bytes = txn.get(b'__metadata__')
        if metadata_bytes is None:
            raise ValueError("Metadata not found in LMDB!")
        metadata = pickle.loads(metadata_bytes)
        logger.info(f"  Metadata keys: {list(metadata.keys())}")
        
        # Check samples
        sample_count = 0
        cursor = txn.cursor()
        for key, value in cursor:
            if key.startswith(b'sample_'):
                sample_count += 1
        
        if sample_count != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, found {sample_count}")
        
        # Load first sample as sanity check
        first_sample_bytes = txn.get(b'sample_000000')
        if first_sample_bytes is None:
            raise ValueError("First sample not found!")
        first_sample = pickle.loads(first_sample_bytes)
        logger.info(f"  First sample keys: {list(first_sample.keys())}")
        logger.info(f"  Position shape: {first_sample['position'].shape}")
        
    env.close()
    logger.info(f"  ✅ Verified {sample_count} samples")


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())


def main():
    parser = argparse.ArgumentParser(description='Convert Zarr dataset to LMDB')
    parser.add_argument('--zarr-path', type=str, required=True,
                       help='Path to input Zarr store')
    parser.add_argument('--lmdb-path', type=str, required=True,
                       help='Path to output LMDB database')
    parser.add_argument('--map-size', type=int, default=50,
                       help='LMDB map size in GB (default: 50)')
    
    args = parser.parse_args()
    
    convert_zarr_to_lmdb(args.zarr_path, args.lmdb_path, args.map_size)


if __name__ == '__main__':
    main()
