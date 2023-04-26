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

def visualise(vars, **kwargs):
    for i, var in enumerate(vars):
        fig, axes = plt.subplots(figsize=(12,9))
        if len(var.shape) > 2 and not var.shape[-1] == 3:
            axes.imshow(var[:,:,-1])
        else:
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

nominal_folder = '4496768/STNS2/27'
image_folder = 'Bagheri/images'
root = os.path.join(os.getcwd(), image_folder)
# Acceptable image file extensions
exts = ['.jpg', '.png']
files = [file for file in os.listdir(root) if file.endswith(tuple(exts))]

photo_z = {"b" : np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524]), 
           "a" : np.array([1, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])}

CSA_KERNEL = np.asarray([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]]) * 1/9

FDSR_TAU_FAST = 3.0
FDSR_TAU_SLOW = 70.0
LPF5_TAU = 25.0

FDSR_K_FAST = np.exp(-1 / FDSR_TAU_FAST)
FDSR_K_SLOW = np.exp(-1 / FDSR_TAU_SLOW)
LPF5_K = np.exp(-1 / LPF5_TAU)

INHIB_KERNEL = np.array([[-1, -1, -1, -1, -1],
                         [-1, 0, 0, 0, -1],
                         [-1, 0, 2, 0, -1],
                         [-1, 0, 0, 0, -1],
                         [-1, -1, -1, -1, -1]])

# Widefield stimuli to check RTC as in Wiederman paper (2008)
def widefield_stimuli():
    PULSE_WIDTH = 5 # Pulse width
    INTERVAL = 10
    TIMESTEPS = (INTERVAL + PULSE_WIDTH) * 20 # Timestep to see adaptation
        
    images = np.ones((240, 320, TIMESTEPS)) * 128
    save_images_root = os.path.join(os.getcwd(), image_folder)
    os.makedirs(save_images_root, exist_ok=True)

    for i in range(TIMESTEPS):
        if i <= TIMESTEPS / 2:
            if not i % (PULSE_WIDTH + INTERVAL):
                for k in range(PULSE_WIDTH):
                    images[:,:,i+k] = 255
        else:
            if not i % (PULSE_WIDTH + INTERVAL):
                for k in range(PULSE_WIDTH):
                    images[:,:,i+k] = 64
    
    images = np.concatenate((np.ones((*images[:,:,-1].shape, 5)) * 128, images), axis=-1)
    
    for i in range(images.shape[-1]):
        cv2.imwrite(os.path.join(save_images_root, 'IMG00' + f'{i+1}' + '.png'), images[:,:,i])
                    
    return images
    
# IIR temporal band-pass filter
def IIR_Filter(b, a, Signal, dbuffer):
    dbuffer[:,:,:-1] = dbuffer[:,:,1:]
    dbuffer[:,:,-1] = np.zeros(dbuffer[:,:,-1].shape)
    
    for k in range(len(b)):
        dbuffer[:,:,k] += (Signal * b[k])
        if k <= (len(b)-2):
            dbuffer[:,:,k+1] = dbuffer[:,:,k+1] - (dbuffer[:,:,0] * a[k+1])
    
    Filtered_Data = dbuffer[:,:,0]
    return Filtered_Data, dbuffer

"""
Initialisations
"""
sf = np.zeros((240, 320, len(files)))
PhotoreceptorOut = np.zeros((34, 46, len(files)))
LMC_Out = np.zeros((34, 46, len(files)))
on_f = np.zeros((34, 46, len(files)))
off_f = np.zeros((34, 46, len(files)))
fdsr_on = np.zeros((34, 46, len(files)))
fdsr_off = np.zeros((34, 46, len(files)))
a_on = np.zeros((34, 46, len(files)))
a_off = np.zeros((34, 46, len(files)))
on_filtered = np.zeros((34, 46, len(files)))
off_filtered = np.zeros((34, 46, len(files)))
delayed_on = np.zeros((34, 46, len(files)))
delayed_off = np.zeros((34, 46, len(files)))
ESTMD_Output = np.zeros((34, 46, len(files)))
RTC_Output = np.zeros((34, 46, len(files)))

"""
Simulation of ESTMD
"""

for t, file in enumerate(files):
    image = cv2.imread(os.path.join(root, file))
    degrees_in_image = 50
    
    # Extract green channel from BGR
    green = image[:,:,1]
    
    image_size = green.shape
    pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
    pixel2PR = int(np.ceil(pixels_per_degree)) # ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling... )
    
    sigma = 1.4 / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixel = sigma * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree
    kernel_size = int(2 * np.ceil(sigma_pixel))
    
    # Spatial filtered through LPF1
    sf[:,:,t] = gaussian_filter(green, radius=kernel_size, sigma=sigma_pixel)
    
    # Downsampled green channel
    DownsampledGreen = cv2.resize(sf[:,:,t], None, fx=1/pixel2PR, fy=1/pixel2PR, interpolation=cv2.INTER_AREA)
    
    try:
        dbuffer1
    except NameError:
        dbuffer1 = np.zeros((*DownsampledGreen.shape, len(photo_z["b"])))
    
    # Photoreceptor output after temporal band-pass filtering
    PhotoreceptorOut[:,:,t], dbuffer1 = IIR_Filter(photo_z["b"], photo_z["a"], DownsampledGreen/255, dbuffer1)
    
    # LMC output after spatial high pass filtering
    LMC_Out[:,:,t] = signal.convolve2d(PhotoreceptorOut[:,:,t], CSA_KERNEL, boundary='symm', mode='same')
    
    # Half-wave rectification
    # Clamp the high pass filtered data to separate the on and off channels
    on_f[:,:,t] = np.maximum(LMC_Out[:,:,t], 0.0)
    off_f[:,:,t] = -np.minimum(LMC_Out[:,:,t], 0.0)
    
    # FDSR Implementation
    if t > 0:
        k_on = np.where((on_f[:,:,t] - on_f[:,:,t-1]) > 0.01, FDSR_K_FAST, FDSR_K_SLOW)
        k_off = np.where((off_f[:,:,t] - off_f[:,:,t-1]) > 0.01, FDSR_K_FAST, FDSR_K_SLOW)
        # Doesn't match Weiderman paper but is the only thing that makes sense
        
        # Apply low-pass filters to on and off channels
        fdsr_on[:,:,t] = ((1.0 - k_on) * on_f[:,:,t]) + (k_on * fdsr_on[:,:,t-1])
        fdsr_off[:,:,t] = ((1.0 - k_off) * off_f[:,:,t]) + (k_off * fdsr_off[:,:,t-1])
        
        # Subtract FDSR from half-wave rectified 
        a_on[:,:,t] = (on_f[:,:,t] - fdsr_on[:,:,t]).clip(min=0)
        a_off[:,:,t] = (off_f[:,:,t] - fdsr_off[:,:,t]).clip(min=0)
        
        # Inhibition is implemented as spatial filter
        # Not true if this is a chemical synapse as this would require delays
        # Half-wave rectification added
        on_filtered[:,:,t] = signal.convolve2d(a_on[:,:,t], INHIB_KERNEL, boundary='symm', mode='same').clip(min=0)
        off_filtered[:,:,t] = signal.convolve2d(a_off[:,:,t], INHIB_KERNEL, boundary='symm', mode='same').clip(min=0)
        
        # Delayed channels for LPF5
        delayed_on[:,:,t] = ((1.0 - LPF5_K) * on_filtered[:,:,t]) + (LPF5_K * delayed_on[:,:,t-1])
        delayed_off[:,:,t] = ((1.0 - LPF5_K) * off_filtered[:,:,t]) + (LPF5_K * delayed_off[:,:,t-1])
        
        # Correlation between channels
        Correlate_ON_OFF = on_filtered[:,:,t] * delayed_off[:,:,t]
        Correlate_OFF_ON = off_filtered[:,:,t] * delayed_on[:,:,t]
        
        ESTMD_Output[:,:,t] = Correlate_ON_OFF + Correlate_OFF_ON
        
        # RTC
        RTC_on = on_filtered[:,:,t] + delayed_off[:,:,t]
        RTC_off = off_filtered[:,:,t] + delayed_on[:,:,t]
        
        RTC_Output[:,:,t] = RTC_on + RTC_off

def _RTC(vars, **kwargs):
    # Initialise ragged nested list
    var_vals = [[[] for i in range(len(j))] for j in vars]
    for v, var in enumerate(vars):
        for i, var_t in enumerate(var):
            for j in range(var_t.shape[-1]):
                # All values in photoreceptor array should be the same
                frame = var_t[:,:,j]
                var_vals[v][i].append(frame[0][0])
            
        fig, ax = plt.subplots(figsize=(12,9))
        for i, val in enumerate(var_vals[v]):
            ax.plot(val)
        ax.set_xlabel('Time (msec)')
        if len(var) > 1:
            ax.legend(labels = ['On', 'Off'])
        ax.get_yaxis().set_visible(False)
        
        RTC_Out_folder = os.path.join(save_folder, 'RTC_images')
        os.makedirs(RTC_Out_folder, exist_ok=True)

        save_fig(os.path.join(RTC_Out_folder, kwargs['title'][v]))
        
    return var_vals

_RTC([[sf], [PhotoreceptorOut], [LMC_Out], [fdsr_on, fdsr_off], [ESTMD_Output], [RTC_Output], [on_filtered, delayed_off], [delayed_on, off_filtered],
      [on_f, off_f], [a_on, a_off]], 
      title=['sf', 'PhotoreceptorOut', 'LMC_Out', 'FDSR', 'ESTMD', 'RTC', 'On_D_OFF', 'Off_D_ON', 'HWR1', 'After FDSR'])

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
    
    fig, ax = plt.subplots(figsize=(12,9))
    ax.plot(T, G, label='log-normal')
    ax.step(t, np.squeeze(y), label='approximation')
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Normalised Amplitude')
    
    ax.legend(fontsize=25)
    save_fig('Time domain to z transform')
    
def __plots():
    visualise([image, green, sf, DownsampledGreen, PhotoreceptorOut, LMC_Out, a_on, a_off, on_filtered, off_filtered, ESTMD_Output, RTC_Output], 
              title=['image', 'green', 'sf', 'DownsampledGreen', 'PhotoreceptorOut', 'LMC_Out', 'After FDSR_on', 'After FDSR_off', 
                     'on_filtered', 'off_filtered', 'ESTMD_Output', 'RTC_Output'])

# continuous(photo_z["b"], photo_z["a"])    
# __plots()