'''
Just-In-Time compilation of the revpref module.
Using Numba to compile the functions in the revpref module.
'''

from .ccei import _warshall_ccei
import numpy as np

try:
    from numba import jit
    _wccei = jit(no_python = True)(_warshall_ccei)
except ImportError:
    jit = None
    pass

def _warshall_ccei_jit(p:np.ndarray, q:np.ndarray):
    return _wccei(p, q)