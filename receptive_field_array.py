"""Recpetive field array for dSTMD to cover receptive field of TSDN
"""

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

def RF_array(mean: list, sigma: list, overlap: float, screen_resolution: list, vf:np.ndarray):
    # Radius of circle = half_width of half maximum
    half_width = np.asarray(sigma) * np.sqrt(2 * np.log(2))
    
    # Find the offset as a function of the overlap
    offset = 2 * (1 - overlap) * half_width
    # Create a grid
    elevation = np.arange(0, screen_resolution[1], offset[1])
    azimuth = np.arange(0, screen_resolution[0], offset[0])
    grid = np.array(list(itertools.product(elevation, azimuth)))
    
    # Re-format grid to convention
    grid[:, [0, 1]] = grid[:, [1, 0]]
    
    grid[:,0] -= (screen_resolution[0] / 2)
    grid[:,1] -= (screen_resolution[1] / 2)
    
    # Find nearest tuple to the mean
    nearest = np.array(min(grid, key=lambda x: math.hypot(x[0] - mean[0], x[1] - mean[1])))
    distance = mean - nearest
    
    points = grid + distance
    
    # Converting to pixels
    pixels_per_degree = np.flip(vf.shape[:-1]) / screen_resolution
    points_degrees2pixels = (points + screen_resolution / 2) * pixels_per_degree
    
    # Extracting TSDN receptive field centers
    hist = np.histogram2d(points_degrees2pixels[:,1], points_degrees2pixels[:,0], vf.shape[:-1])[0]
    TSDN_rf = vf[...,0].copy()
    TSDN_rf[TSDN_rf <= 0.5] = 0 # only take within 50% amplitude
    
    # Remove points not within TSDN receptive field
    TSDN_overlay = hist * TSDN_rf
    TSDN_coordinates = np.flip(np.stack((np.where(TSDN_overlay > 0))).T, axis=1)
    
    # Convert back to degrees
    coordinates_pixels2degrees = (TSDN_coordinates / pixels_per_degree ) - screen_resolution / 2
    coordinates_pixels2degrees[:,1] = np.abs(coordinates_pixels2degrees[:,1])
    
    fig, axes = plt.subplots(dpi=200)
    axes.scatter(coordinates_pixels2degrees[:,0], coordinates_pixels2degrees[:,1], s=5)
    x_extent, y_extent = screen_resolution / 2
    axes.imshow(TSDN_rf, extent=[-x_extent, x_extent, -y_extent, y_extent])
    axes.grid()
    axes.set_xlabel('Azimuth [$^\circ$]')
    axes.set_ylabel('Elevation [$^\circ$]')
    
    return coordinates_pixels2degrees