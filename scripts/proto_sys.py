import tensorflow as tf
import sionna
try:
    from sionna.sys import PHYAbstraction
    print("PHYAbstraction imported")

    # Try to instantiate
    # It requires parameters?
    try:
        phy_abs = PHYAbstraction(
            mcs_tables='38.214', # default
            numerical_cbs_handling=True
        )
        print("PHYAbstraction instantiated")
        
        # Try call
        # Input: snr (dB), mcs_index, block_size
        batch_size = 5
        snr = tf.random.uniform((batch_size,), 0, 30)
        mcs = tf.random.uniform((batch_size,), 0, 27, dtype=tf.int32)
        size = tf.fill((batch_size,), 1000)
        
        bler = phy_abs(snr, mcs, size)
        print("BLER computed:", bler.numpy())
        
    except Exception as e:
        print(f"Usage failed: {e}")
        
except ImportError:
    print("PHYAbstraction not found in sionna.sys")
