"""Recpetive field array for dSTMD to cover receptive field of TSDN
"""

import itertools
import math
import numpy as np

def RF_array(mean: list, sigma: list, overlap: float, screen_resolution: list):
    # Radius of circle = half_width of half maximum
    half_width = np.asarray(sigma) * np.sqrt(2 * np.log(2))
    
    # Find the offset as a function of the overlap
    offset = 2 * (1 - overlap) * half_width
    # Create a grid
    grid = np.array(list(itertools.product(np.arange(0, screen_resolution[1], offset[1]), np.arange(0, screen_resolution[0], offset[0]))))
    
    # Re-format grid to convention
    grid[:, [0, 1]] = grid[:, [1, 0]]
    
    grid[:,0] -= (screen_resolution[0] / 2)
    grid[:,1] -= (screen_resolution[1] / 2)
    
    # Find nearest tuple to the mean
    nearest = np.array(min(grid, key=lambda x: math.hypot(x[0] - mean[0], x[1] - mean[1])))
    distance = mean - nearest
    
    return grid + distance