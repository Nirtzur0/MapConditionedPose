
import tensorflow as tf
import numpy as np
from src.data_generation.features import PHYFAPIFeatureExtractor, MACRRCFeatureExtractor, RTLayerFeatures, compute_rsrp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_dimensions():
    print("=== Debugging Feature Dimension Handling ===")
    
    # 1. Setup Mock Inputs (Batch=32, Sites=1)
    batch_size = 32
    num_sites = 1
    num_rx_ant = 1
    num_tx_ant = 1
    num_freq = 128 # Must be > 100 to satisfy default pilot indices
    
    print(f"Config: Batch={batch_size}, Sites={num_sites}")
    
    # Mock Channel Matrix: [Batch, Sites, RxAnt, TxAnt, Freq]
    # Representing output after RT transpose
    # Shape: [32, 1, 1, 1, 64]
    channel_matrix_np = np.random.randn(batch_size, num_sites, num_rx_ant, num_tx_ant, num_freq) + \
                        1j * np.random.randn(batch_size, num_sites, num_rx_ant, num_tx_ant, num_freq)
    channel_matrix = tf.convert_to_tensor(channel_matrix_np, dtype=tf.complex64)
    print(f"Channel Matrix Shape: {channel_matrix.shape}")
    
    # Mock RT Features (needed for PHY extract)
    rt_features = RTLayerFeatures(
        path_gains=np.zeros((batch_size, num_sites, 3)), # Dummy
        path_delays=np.zeros((batch_size, num_sites, 3)),
        path_aoa_azimuth=None, path_aoa_elevation=None,
        path_aod_azimuth=None, path_aod_elevation=None,
        path_doppler=None,
        rms_delay_spread=None, rms_angular_spread=None,
        k_factor=None, num_paths=None,
        carrier_frequency_hz=3.5e9, bandwidth_hz=100e6, is_mock=False
    )
    
    # 2. Test compute_rsrp direct
    pilot_ind = tf.range(0, num_freq, 4)
    rsrp = compute_rsrp(channel_matrix, 0, pilot_ind, use_tf=True)
    print(f"compute_rsrp Output Shape: {rsrp.shape}")
    # Expectation: [32, 1]
    
    # Mock Channel Matrix (Created but not passed to PHY to simulate fallback path)
    # channel_matrix = ... 
    
    # 3. Test PHY Extraction (Fallback path using RT gains)
    phy_extractor = PHYFAPIFeatureExtractor()
    # Passing channel_matrix=None to trigger fallback RSRP calc
    phy_features = phy_extractor.extract(rt_features, channel_matrix=None)
    print(f"PHY Features RSRP Shape: {phy_features.rsrp.shape}")
    # Expectation: [32, 1, 1] (Unsqueezed)
    
    # 4. Test MAC Extraction (The crash site)
    mac_extractor = MACRRCFeatureExtractor()
    ue_pos = np.zeros((batch_size, 3))
    site_pos = np.zeros((num_sites, 3))
    cell_ids = np.array([101]) # Size 1
    
    try:
        mac_features = mac_extractor.extract(
            phy_features=phy_features,
            ue_positions=ue_pos,
            site_positions=site_pos,
            cell_ids=cell_ids
        )
        print("MAC Extraction Success!")
        print(f"Serving Cell IDs: {mac_features.serving_cell_id}")
    except Exception as e:
        print(f"MAC Extraction Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dimensions()
