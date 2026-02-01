"""
Quick test script to verify all imports work correctly
"""
import sys

print("Testing imports...")

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print("✓ TensorFlow imported successfully")
    print(f"  TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print(f"  NumPy version: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import scipy.io
    print("✓ SciPy imported successfully")
except Exception as e:
    print(f"✗ SciPy import failed: {e}")
    sys.exit(1)

try:
    from utilities import neural_net, Navier_Stokes_3D, tf_session
    print("✓ utilities module imported successfully")
except Exception as e:
    print(f"✗ utilities import failed: {e}")
    sys.exit(1)

try:
    from pyevtk.hl import gridToVTK
    print("✓ pyevtk imported successfully")
except ImportError:
    print("⚠ pyevtk not available (optional for VTK output)")

try:
    import pyvista as pv
    print("✓ pyvista imported successfully")
except ImportError:
    print("⚠ pyvista not available (optional for VTK output)")

print("\n✓ All critical imports successful!")
print("You can now run the main scripts.")

