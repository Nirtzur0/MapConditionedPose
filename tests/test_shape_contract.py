import numpy as np

from src.data_generation.shape_contract import normalize_channel_matrix


def test_normalize_channel_matrix_rank5():
    data = np.zeros((2, 4, 3, 2, 16))
    cm = normalize_channel_matrix(data, strict=True)
    assert cm.data.shape == (2, 1, 4, 3, 2, 16)
    assert cm.batch == 2
    assert cm.rx == 1
    assert cm.rx_ant == 4
    assert cm.cells == 3
    assert cm.tx_ant == 2
    assert cm.freq == 16


def test_normalize_channel_matrix_rank6():
    data = np.zeros((2, 1, 4, 3, 2, 16))
    cm = normalize_channel_matrix(data, strict=True)
    assert cm.data.shape == (2, 1, 4, 3, 2, 16)
