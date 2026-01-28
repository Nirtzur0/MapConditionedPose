import numpy as np

from src.config.feature_schema import MACFeatureIndex
from src.datasets.lmdb_dataset import LMDBRadioLocalizationDataset


def test_multi_site_serving_and_neighbors_mapping():
    dataset = LMDBRadioLocalizationDataset(
        lmdb_path="dummy",
        split="train",
        map_resolution=1.0,
        scene_extent=512,
        normalize=False,
        max_cells=4,
    )

    sample = {
        "rt_features": {"toa": np.array([1.0, 1.0, 1.0], dtype=np.float32)},
        "phy_features": {"rsrp": np.array([-70.0, -60.0, -90.0], dtype=np.float32)},
        "mac_features": {
            "serving_cell_id": np.array([202], dtype=np.int32),
            "neighbor_cell_ids": np.array([[101, 303]], dtype=np.int32),
            "timing_advance": np.array([12.0], dtype=np.float32),
            "dl_throughput_mbps": np.array([50.0], dtype=np.float32),
            "bler": np.array([0.2], dtype=np.float32),
        },
        "scene_metadata": {
            "sites": [
                {"cell_id": 101},
                {"cell_id": 202},
                {"cell_id": 303},
            ]
        },
        "timestamp": 0.0,
    }

    processed = dataset._process_measurements(sample)
    mac = processed["mac_features"].numpy()

    serving_idx = 1  # cell_id 202 -> slot 1
    neighbor1_idx = 0  # cell_id 101 -> slot 0
    neighbor2_idx = 2  # cell_id 303 -> slot 2

    assert mac[serving_idx, MACFeatureIndex.SERVING_CELL_ID] == 1.0
    assert mac[neighbor1_idx, MACFeatureIndex.NEIGHBOR_CELL_ID_1] == 1.0
    assert mac[neighbor2_idx, MACFeatureIndex.NEIGHBOR_CELL_ID_2] == 1.0
    assert mac[serving_idx, MACFeatureIndex.TIMING_ADVANCE] == 12.0
    assert mac[serving_idx, MACFeatureIndex.DL_THROUGHPUT] == 50.0
    assert np.isclose(mac[serving_idx, MACFeatureIndex.BLER], 0.2)
