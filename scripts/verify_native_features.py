
import tensorflow as tf
import numpy as np
from src.data_generation.native_features import SionnaNativeKPIExtractor
from dataclasses import dataclass

# Mock Paths Object
@dataclass
class MockPaths:
    a: tf.Tensor
    tau: tf.Tensor
    phi_r: tf.Tensor
    theta_r: tf.Tensor
    phi_t: tf.Tensor
    theta_t: tf.Tensor

def run_verification():
    print("Initializing Extractor...")
    extractor = SionnaNativeKPIExtractor()
    
    batch_size = 2
    num_cells = 3
    num_paths = 5
    
    print("--- Verifying RT Extraction ---")
    # Mock Inputs [Sources, Targets, Paths, RxA, TxA, 1, 1]
    # S=Cells, T=Batch
    a = tf.complex(tf.random.normal((num_cells, batch_size, num_paths, 1, 1, 1, 1)), 
                   tf.random.normal((num_cells, batch_size, num_paths, 1, 1, 1, 1)))
    tau = tf.abs(tf.random.normal((num_cells, batch_size, num_paths))) * 1e-6
    angles = tf.random.uniform((num_cells, batch_size, num_paths), 0, 2*np.pi)
    
    paths = MockPaths(a=a, tau=tau, phi_r=angles, theta_r=angles, phi_t=angles, theta_t=angles)
    
    rt = extractor.extract_rt(paths, batch_size)
    for k, v in rt.items():
        print(f"  {k}: shape={v.shape}")
        
    # Check expected shapes [Batch, Cells]
    assert rt['toa'].shape == (batch_size, num_cells)
    
    print("\n--- Verifying PHY Extraction ---")
    # Channel Matrix: [Batch, RxAnt, Cells, TxAnt, Freq]
    freqs = 32
    h = tf.complex(tf.random.normal((batch_size, 1, num_cells, 1, freqs)),
                   tf.random.normal((batch_size, 1, num_cells, 1, freqs)))
                   
    phy = extractor.extract_phy(h, batch_size)
    for k, v in phy.items():
        print(f"  {k}: shape={v.shape}")
        
    assert phy['rsrp'].shape == (batch_size, num_cells)
    
    print("\nVerification Passed!")

if __name__ == "__main__":
    run_verification()
