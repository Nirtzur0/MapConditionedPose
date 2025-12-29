import sys
import os
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SionnaVerify")

def verify_sionna_upgrade():
    logger.info("Verifying Sionna Upgrade...")
    
    try:
        import sionna
        logger.info(f"Sionna version: {sionna.__version__}")
    except ImportError:
        logger.error("Failed to import sionna!")
        return False

    # Check Key Modules
    try:
        import sionna.sys
        logger.info("✅ sionna.sys found")
    except ImportError:
        logger.error("❌ sionna.sys NOT found")
        
    try:
        from sionna.phy.channel import cir_to_ofdm_channel
        logger.info("✅ sionna.phy.channel.cir_to_ofdm_channel found")
    except ImportError:
         logger.error("❌ cir_to_ofdm_channel NOT found")

    # Run a simple check
    try:
        h = tf.eye(2, dtype=tf.complex64)
    except Exception as e:
        logger.error(f"❌ TF check failed: {e}")

    return True

if __name__ == "__main__":
    verify_sionna_upgrade()
