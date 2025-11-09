# lib/utils.py
"""
Description: convert NumPy to Python types.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Nov 09
Version: 1.0.0 
"""
import numpy as np

def to_python_types(obj):
    """Recursively convert NumPy scalar types to native Python types for MongoDB compatibility."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj