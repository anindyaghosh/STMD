"""Fit Gaussians to different neuron types
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

vf = np.zeros((1440, 2560, 3))
screen_resolution = np.array([155, 138])

pixels_per_degree = np.min(np.flip(vf.shape[:-1]) / screen_resolution)

neurons = {"TSDN" : {"sigma_vals":12.74, "centre":(0, 45)}, 
           "dSTMD" : {"sigma_vals":3, "centre":(0, 55)}, 
           "wSTMD" : {"sigma_vals":12.74, "centre":(0, 55)}}

def rf_imshow(index):
    fig, axes = plt.subplots(figsize=(12,9))
    x_extent = screen_resolution[0]
    y_extent = int(1/pixels_per_degree * vf.shape[0]/2)
    axes.imshow(vf[:,:,index], extent=[-x_extent, x_extent, -y_extent, y_extent])
    axes.grid()
    axes.set_xlabel('Azimuth [$^\circ$]')
    axes.set_ylabel('Elevation [$^\circ$]')
    # hide_axes(axes)

def hide_axes(axis):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

def matlab_style_gauss2D(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian', [shape], [sigma])
    """
    m, n = [(ss-1)/2 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def calc_kernel_size(sigma: float):
    # Sigma is in degrees
    sigma_pixel = sigma * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree
    kernel_size = int(2 * np.ceil(sigma_pixel))
    
    return kernel_size, sigma_pixel

def np_bivariate_normal_pdf(domain, mean : tuple, variance):
    X = np.arange(-domain+mean[0], domain+mean[0], variance)
    Y = np.arange(-domain+mean[1], domain+mean[1], variance)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
    return X+mean[0], Y+mean[1], Z

def plt_plot_bivariate_normal_pdf(x, y, z):
    fig, axes = plt.subplots(figsize=(12, 6))
    h = axes.imshow(z)
    # fig = plt.figure()
    # axes = fig.add_subplot(projection='3d')
    # axes.plot_surface(x, y, z, 
    #                 cmap=cm.coolwarm,
    #                 linewidth=0, 
    #                 antialiased=True)
    # axes.set_zlabel('z')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.colorbar(h)
    plt.show()

def calc_centre_coordinates(location : tuple):
    # Location is in degrees (x, y) -> (column, row) where x and y are elements of (-180, 180)
    location_in_pixels = np.asarray(location) * pixels_per_degree
    vf_centre = np.flip(vf.shape[:-1])/2
    
    x = int(vf_centre[0] + location_in_pixels[0])
    y = int(vf_centre[1] - location_in_pixels[1])
    
    return (x, y)

def composite_receptive_field(neuron_type, index):
    # Get kernel size and sigma in pixels
    kernel_size, sigma_pixel = calc_kernel_size(neurons[neuron_type]["sigma_vals"])
    
    # Get Gaussian kernel
    H = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma_pixel)
    
    # Receptive field
    loc_centre = calc_centre_coordinates(neurons[neuron_type]["centre"])
    kernel_centre = tuple([int(l / 2) for l in H.shape])
    
    
    vf[loc_centre[1]-kernel_centre[1]:loc_centre[1]+kernel_centre[1],
       loc_centre[0]-kernel_centre[0]:loc_centre[0]+kernel_centre[0],index] = H
    
    if neuron_type == 'wSTMD':
        vf[:,:loc_centre[0],index] = 0
    
    rf_imshow(index)
    
    return vf
    
for i, neuron in enumerate(neurons.keys()):
    vf = composite_receptive_field(neuron, i)
    
# plt_plot_bivariate_normal_pdf(*np_bivariate_normal_pdf(20, (4, 2), .25))