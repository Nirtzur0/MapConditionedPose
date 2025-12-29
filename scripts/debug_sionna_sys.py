import numpy as np
import tensorflow as tf
import sionna
import sionna.sys

print("Sionna Version:", sionna.__version__)

# Inspect PHYAbstraction
try:
    print("\n--- defaults for sionna.sys.PHYAbstraction ---")
    # It might be a class
    if hasattr(sionna.sys, 'PHYAbstraction'):
        print(help(sionna.sys.PHYAbstraction))
    else:
        print("PHYAbstraction not found in sionna.sys")
except Exception as e:
    print(f"Error inspecting PHYAbstraction: {e}")

# Inspect LinkAdaptation
try:
    print("\n--- defaults for sionna.sys.LinkAdaptation ---")
    if hasattr(sionna.sys, 'LinkAdaptation'):
        # It's abstract, maybe inner loop?
        pass
    
    if hasattr(sionna.sys, 'InnerLoopLinkAdaptation'):
         # print(help(sionna.sys.InnerLoopLinkAdaptation)) # might be too long
         pass
except Exception as e:
    print(f"Error inspecting LinkAdaptation: {e}")
    
# Debug Broadcasting Error
print("\n--- Debugging Broadcasting ---")
try:
    # Hyp: u_pos=(32,1,3), serv_pos=(2,32,3)
    u_pos = np.zeros((32, 1, 3))
    serv_pos = np.zeros((2, 32, 3))
    print(f"Shapes: u_pos={u_pos.shape}, serv_pos={serv_pos.shape}")
    dist = np.linalg.norm(u_pos - serv_pos, axis=-1)
    print("Success:", dist.shape)
except Exception as e:
    print("Failure:", e)

# How could serv_pos get that shape?
# serving_pos = site_positions[best_idx]
# if site_positions is (N_sites, 3) -> (2, 3)
# best_idx must be (2, 32) ??
# best_cell_idx = np.argmax(rsrp, axis=-1)
# if rsrp is (32, 1, 2), argmax is (32, 1). 
# site_positions[(32,1)] -> (32, 1, 3). Correct.

# What if rsrp was transposed?
# if rsrp is (Rx=1, Batch=32, Cells=2)? -> (1, 32, 2)
# argmax(-1) -> (1, 32).
# site_positions[(1, 32)] -> (1, 32, 3).
# u_pos is (32, 1, 3).
# (1, 32, 3) - (32, 1, 3) -> (32, 32, 3). Works (broadcasting).

# What if rsrp is (Cells=2, Batch=32, Rx=1)?
# argmax -> (Cells?, Batch?) No, argmax over cells.
# If axis=-1 is not cells?

# Let's verify standard rsrp shape from features.py
