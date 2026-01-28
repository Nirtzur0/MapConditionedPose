"""
Shared feature schema for generation, dataset packing, and physics loss.

This is the single source of truth for:
- Feature vector indices used by the dataset/model
- Storage key names used during generation
- Radio map channel ordering for physics loss
"""

from enum import IntEnum
from typing import Dict, Tuple


class RTFeatureIndex(IntEnum):
    TOA = 0
    MEAN_PATH_GAIN = 1  # unused (removed from model inputs)
    MAX_PATH_GAIN = 2  # unused (removed from model inputs)
    MEAN_PATH_DELAY = 3  # unused (removed from model inputs)
    NUM_PATHS = 4  # unused (removed from model inputs)
    RMS_DELAY_SPREAD = 5
    RMS_ANGULAR_SPREAD = 6
    TOTAL_POWER = 7
    N_SIGNIFICANT_PATHS = 8
    DELAY_RANGE = 9  # unused (removed from model inputs)
    DOMINANT_PATH_GAIN = 10  # unused (removed from model inputs)
    DOMINANT_PATH_DELAY = 11  # unused (removed from model inputs)
    IS_NLOS = 12
    DOPPLER_SPREAD = 13
    COHERENCE_TIME = 14
    RESERVED_15 = 15


class PHYFeatureIndex(IntEnum):
    RSRP = 0
    RSRQ = 1
    SINR = 2
    CQI = 3
    RI = 4
    PMI = 5
    L1_RSRP = 6  # unused (removed from model inputs)
    BEST_BEAM_ID = 7


class MACFeatureIndex(IntEnum):
    SERVING_CELL_ID = 0
    NEIGHBOR_CELL_ID_1 = 1
    NEIGHBOR_CELL_ID_2 = 2
    TIMING_ADVANCE = 3
    DL_THROUGHPUT = 4
    BLER = 5


RT_FEATURE_DIM = 16
PHY_FEATURE_DIM = 8
MAC_FEATURE_DIM = 6

RADIO_MAP_CHANNELS: Tuple[str, ...] = (
    "rsrp",
    "rsrq",
    "sinr",
    "cqi",
    "throughput",
)

PHYSICS_OBSERVED_FEATURES: Tuple[str, ...] = RADIO_MAP_CHANNELS

RT_KEYS: Dict[str, str] = {
    "toa": "rt/toa",
    "path_gains": "rt/path_gains",
    "path_delays": "rt/path_delays",
    "num_paths": "rt/num_paths",
    "rms_delay_spread": "rt/rms_delay_spread",
    "rms_angular_spread": "rt/rms_angular_spread",
    "doppler_spread": "rt/doppler_spread",
    "coherence_time": "rt/coherence_time",
    "k_factor": "rt/k_factor",
    "is_nlos": "rt/is_nlos",
}

PHY_KEYS: Dict[str, str] = {
    "rsrp": "phy_fapi/rsrp",
    "rsrq": "phy_fapi/rsrq",
    "sinr": "phy_fapi/sinr",
    "cqi": "phy_fapi/cqi",
    "ri": "phy_fapi/ri",
    "pmi": "phy_fapi/pmi",
    "l1_rsrp_beams": "phy_fapi/l1_rsrp_beams",
    "best_beam_ids": "phy_fapi/best_beam_ids",
    "channel_matrix": "phy_fapi/channel_matrix",
    "cfr_magnitude": "phy_fapi/cfr_magnitude",
    "cfr_phase": "phy_fapi/cfr_phase",
    "capacity_mbps": "phy_fapi/capacity_mbps",
    "condition_number": "phy_fapi/condition_number",
}

MAC_KEYS: Dict[str, str] = {
    "serving_cell_id": "mac_rrc/serving_cell_id",
    "neighbor_cell_ids": "mac_rrc/neighbor_cell_ids",
    "timing_advance": "mac_rrc/timing_advance",
    "tracking_area_code": "mac_rrc/tracking_area_code",
    "dl_throughput_mbps": "mac_rrc/dl_throughput_mbps",
    "ul_throughput_mbps": "mac_rrc/ul_throughput_mbps",
    "bler": "mac_rrc/bler",
    "handover_events": "mac_rrc/handover_events",
    "time_since_handover": "mac_rrc/time_since_handover",
}
