"""
ESTMD model as per Bagheri and Wiederman
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy import signal
import time

"""
Helper functions
"""

save_folder = os.path.join(os.getcwd(), 'Bagheri')

def visualise(vars, labels=None, **kwargs):
    for i, var in enumerate(vars):
        fig, axes = plt.subplots(figsize=(15,10))
        axes.imshow(var)
                
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        
        save_fig(kwargs['title'][i])

# Helper function to hide axes ticks etc on axis
def hide_axes(axis):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

# Helper function to save out images from different stages    
def save_fig(title):
    save_out = os.path.join(os.getcwd(), 'figs', time.strftime("%Y_%m_%d"), save_folder)
    # save_out = os.path.join(os.getcwd(), 'figs', 'test', save_folder)
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title))
    plt.close()

images = cv2.imread(os.path.join(os.getcwd(), '4496768/STNS1/26', 'IMG00896.jpg'))
degrees_in_image = 50

# Extract green channel from BGR
green = images[:,:,1]

image_size = green.shape
pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
pixel2PR = int(np.ceil(pixels_per_degree)) # ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling... )

sigma = 1.4 / (2 * np.sqrt(2 * np.log(2)))
sigma_pixel = sigma * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree
kernel_size = int(2 * np.ceil(sigma_pixel))

# Spatial filtered through LPF1
sf = gaussian_filter(green, radius=kernel_size, sigma=sigma_pixel)

# Downsampled green channel
DownsampledGreen = sf[::pixel2PR, ::pixel2PR]
# DownsampledGreen = cv2.resize(sf, None, fx=1/pixel2PR, fy=1/pixel2PR, interpolation=cv2.INTER_AREA)

b = np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524])
a = np.array([1, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])

def IIR_Filter(b, a, Signal, dbuffer):
    for k in range(len(b)-1):
        dbuffer[:,:,k]=dbuffer[:,:,k+1];
    dbuffer[:,:,-1]=np.zeros(dbuffer[:,:,-1].shape);
    for k in range(len(b)):
        dbuffer[:,:,k]=dbuffer[:,:,k]+Signal*b[k];
    for k in range(len(b)-1):
        dbuffer[:,:,k+1]=dbuffer[:,:,k+1]-dbuffer[:,:,0]*a[k+1];
    Filtered_Data=dbuffer[:,:,0]
    return Filtered_Data, dbuffer

dbuffer1 = np.zeros((35,46,len(b)))
for i in range(len(b)):
    image_z, dbuffer1 = IIR_Filter(b, a, DownsampledGreen/255, dbuffer1)

def continuous(b, a):
    # Difference of Lognormals - Continuous filter
    # PDF h(t) = a exp (-(ln (t/tp)) ** 2 /2(s **2)), t [ms]
    T, G = [], []
    pk_t = 11.2
    for t in np.arange(60 / pk_t, step=0.001):
        T.append(t / (1000) * pk_t)
        G.append((1.06 * np.exp((-(np.log(t / 1.01)) ** 2) / 0.0776)) + \
                 (-0.167 * np.exp((-(np.log(t / 1.75)) ** 2) / 0.238)))
            
    # Approximation of log-normals
    t, y = signal.dimpulse((b, a, 1/1000))
    
    fig, ax = plt.subplots()
    ax.plot(T, G, label='log-normal')
    ax.step(t, np.squeeze(y), label='approximation')
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Normalised Amplitude')
    
    ax.legend()

def __plots():
    visualise([images, green, sf, DownsampledGreen, image_z], title=['images', 'green', 'sf', 'DownsampledGreen', 'image_z'])
    
__plots()