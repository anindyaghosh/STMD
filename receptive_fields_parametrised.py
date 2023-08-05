"""Receptive fields
"""

import matplotlib.pyplot as plt
import numpy as np

vf_resolution = (1440, 2560)
vf = np.zeros((*vf_resolution, 3))
screen_resolution = np.array([155, 138]) # in degrees

pixels_per_degree = np.flip(vf_resolution) / screen_resolution

neurons = {"TSDN" : {"sigma_vals":12.74, "centre":(0, 45)}, 
           "dSTMD" : {"sigma_vals":3, "centre":(0, 55)}, 
           "wSTMD" : {"sigma_vals":12.74, "centre":(0, 55)}}

def calc_pixel_size(value: tuple):
    # Sigma is in degrees
    return value * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree

def calc_centre_coordinates(location : tuple):
    # Location is in degrees (x, y) -> (column, row) where x and y are elements of (-180, 180)
    location_in_pixels = location * pixels_per_degree
    vf_centre = np.flip(vf_resolution)/2
    
    x = int(vf_centre[0] + location_in_pixels[0])
    y = int(vf_centre[1] - location_in_pixels[1])
    
    return (x, y)

def rf_imshow(rf):
    fig, axes = plt.subplots(figsize=(12,9))
    x_extent, y_extent = screen_resolution / 2
    img = axes.imshow(rf, extent=[-x_extent, x_extent, -y_extent, y_extent])
    axes.grid()
    axes.set_xlabel('Azimuth [$^\circ$]')
    axes.set_ylabel('Elevation [$^\circ$]')
    plt.colorbar(img)

def gaussian(mean: tuple, sigma: tuple):
    """For the general form of the Gaussian function, the coefficient A is the height of the peak and (x0, y0) is the center of the Gaussian blob.

    Parameters
    ----------
    mean: tuple
        mean in x and y directions (in degrees).
    sigma: tuple
        sigma in x and y directions (in degrees).

    Returns
    -------
    Receptive field.

    """
    
    if isinstance(sigma, float):
        sigma = (sigma, sigma)
    
    sigma_pixels = calc_pixel_size(sigma)
    mean_pixels = calc_centre_coordinates(mean)
    
    A = 1
    x0, y0 = mean_pixels
    sigma_X, sigma_Y = sigma_pixels
    X, Y = np.meshgrid(np.arange(vf_resolution[1]), np.arange(vf_resolution[0]))
    
    # theta is for rotation of Gaussian blob
    theta = 0
    
    a = np.cos(theta)**2 / (2 * sigma_X ** 2) + np.sin(theta)**2 / (2 * sigma_Y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_X ** 2) + np.sin(2 * theta) / (4 * sigma_Y ** 2)
    c = np.sin(theta)**2 / (2 * sigma_X ** 2) + np.cos(theta)**2 / (2 * sigma_Y ** 2)
    
    Z = A * np.exp(-(a * (X - x0)**2 + 2*b*(X - x0)*(Y - y0) + c*(Y - y0)**2))
    
    if neuron == "wSTMD":
        Z[:,:x0] = 0
        
    return Z
    
for i, neuron in enumerate(neurons.keys()):
    vf[...,i] += gaussian(neurons[neuron]["centre"], neurons[neuron]["sigma_vals"])
    rf_imshow(vf[...,i])