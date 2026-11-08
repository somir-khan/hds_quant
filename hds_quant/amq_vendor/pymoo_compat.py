"""Preserve pymoo 0.6.1.3 with NumPy versions removing the row_stack alias."""
import numpy as np
from pymoo.optimize import minimize as original_minimize


def minimize(*args, **kwargs):
    # row_stack was precisely an alias for vstack. Restore only during this
    # single-threaded search call, without editing NumPy or pymoo on disk.
    missing = not hasattr(np, 'row_stack')
    if missing:
        np.row_stack = np.vstack
    try:
        return original_minimize(*args, **kwargs)
    finally:
        if missing:
            del np.row_stack
