"""
ESTMD model as per Bagheri and Wiederman
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from scipy import signal
import scipy.io
import time

import naturalistic_noise
from receptive_fields_parametrised import receptive_fields as rf

"""
Helper functions
"""

# For quick shell script
import argparse
parser = argparse.ArgumentParser()
# parser.add_argument('-nf', '--nominal_folder', type=str)
parser.add_argument('-v', '--value', type=float)
parser.add_argument('-f', '--folder', type=str)

args = parser.parse_args()

save_folder = os.path.join(os.getcwd(), 'Bagheri')

def visualise(vars, sub_folder=None, **kwargs):
    for i, var in enumerate(vars):
        fig, axes = plt.subplots(figsize=(48,36))
        if len(var.shape) > 2 and not var.shape[-1] == 3:
            axes.imshow(var[:,:,-1])
        elif len(var.shape) == 2:
            axes.imshow(var, cmap='gray')
        else:
            var = cv2.cvtColor(var, cv2.COLOR_BGR2RGB)
            axes.imshow(var)
            if kwargs['rf']:
                axes.contour(vf[...,0], levels=[0.5])
        hide_axes(axes)
        
        save_fig(kwargs['title'][i], sub_folder)

# Helper function to hide axes ticks etc on axis
def hide_axes(axis):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

# Helper function to save out images from different stages
def save_fig(title, sub_folder=None):
    if sub_folder is None:
        save_out = os.path.join(os.getcwd(), 'figs', time.strftime("%Y_%m_%d"), save_folder)
    else:
        save_out = os.path.join(os.getcwd(), 'figs', time.strftime("%Y_%m_%d"), save_folder, sub_folder)
    # save_out = os.path.join(os.getcwd(), 'figs', 'test', save_folder)
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title), bbox_inches='tight', pad_inches=0)
    plt.close()

def naming_convention(i):
    # zfill pads string with zeros from leading edge until len(string) = 6
    return 'IMG' + str(i).zfill(6)

nominal_folder = '4496768/STNS3/28' # args.nominal_folder.rstrip() # '4496768/STNS3/28'
image_folder = 'Bagheri/images'
tuning_folder = 'Bagheri/Tuning'
botanic_folder = 'Bagheri/botanic'
EMD_folder = 'Bagheri/EMD'
root = os.path.join(os.getcwd(), tuning_folder) # Has to change between image_folder and tuning_folder for tuning experiments
os.makedirs(root, exist_ok=True)
# Acceptable image file extensions
exts = ['.jpg', '.png']
files = [file for file in os.listdir(root) if (file.endswith(tuple(exts)) and 'IMG' in file)]

photo_z = {"b" : np.array([0, 0.0001, -0.0011, 0.0052, -0.0170, 0.0439, -0.0574, 0.1789, -0.1524]), 
            "a" : np.array([1, -4.3331, 8.6847, -10.7116, 9.0004, -5.3058, 2.1448, -0.5418, 0.0651])}

"""
TAU for FDSR and LPF5, z-transforms
"""
Ts = 0.001
LPF5_TAU = 25 * Ts
LPF_5 = {"b" : np.array([1 / (1+2*LPF5_TAU/Ts), 1 / (1+2*LPF5_TAU/Ts)]), 
         "a" : np.array([1, (1-2*LPF5_TAU/Ts) / (1+2*LPF5_TAU/Ts)])}

LPFHR_TAU = 40 * Ts

LPF_HR = {"b" : np.array([1 / (1+2*LPFHR_TAU/Ts), 1 / (1+2*LPFHR_TAU/Ts)]), 
          "a" : np.array([1, (1-2*LPFHR_TAU/Ts) / (1+2*LPFHR_TAU/Ts)])}

CSA_KERNEL = np.asarray([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]]) * 1/9

FDSR_TAU_FAST_ON = 0.5/50 * Ts*1000
FDSR_TAU_FAST_OFF = 0.5/50 * Ts*1000

FDSR_TAU_SLOW = 5.0/50 * Ts*1000

FDSR_K_FAST_ON = np.exp(-1*Ts / FDSR_TAU_FAST_ON)
FDSR_K_FAST_OFF = np.exp(-1*Ts / FDSR_TAU_FAST_OFF)

FDSR_K_SLOW = np.exp(-1*Ts / FDSR_TAU_SLOW)

LPF5_K = np.exp(-1*0.05 / LPF5_TAU)

INHIB_KERNEL = 1.2 * np.array([[-1, -1, -1, -1, -1],
                                [-1, 0, 0, 0, -1],
                                [-1, 0, 2, 0, -1],
                                [-1, 0, 0, 0, -1],
                                [-1, -1, -1, -1, -1]])

# 1D kernel
# INHIB_KERNEL = 1.2 * np.array([[-8],
#                           [0],
#                           [2],
#                           [0],
#                           [-8]])

# INHIB_KERNEL = np.array([[-1, -1, -1, -1, -1],
#                           [-1, -1, -1, -1, -1],
#                           [-1, -1, 2, -1, -1],
#                           [-1, -1, -1, -1, -1],
#                           [-1, -1, -1, -1, -1]])

if tuning_folder in root:
    desired_resolution = (72, 72)
elif EMD_folder in root:
    desired_resolution = (138, 155)
elif nominal_folder in root:
    desired_resolution = (None, 50)
else:
    desired_resolution = (None, 360)

degree_of_separation = None # args.value # degrees

def create_EMD_tests():
    # image = naturalistic_noise.fourier()
    # image = cv2.imread('fourier.png')
    # image = cv2.imread('RandomImageLarge.png')
    image = cv2.imread('sparse_field_uniform.png')
    # image = np.ones(image.shape) * 255
    image = cv2.blur(image, (5,5))
    
    bg_dims = image[1000:1400,2000:4594].copy()
    
    # Target size definitions
    pixels_per_degree = bg_dims.shape[1]/360
    
    # Conforming to desired visual field
    pixels_to_keep = (np.array(desired_resolution) * pixels_per_degree).astype(int)
    
    target_size = round(2.1 * pixels_per_degree) # 2.2 degrees is optimal target size
    start = (1100, 0) # 250
    
    # Adjust starting row
    lst = list(start)
    lst[0] -= 1000
    start = tuple(lst)
    
    # Calculate velocity of image
    velocity = (160/1000) * pixels_per_degree # 160 degrees/second gives optimal latency of 29 ms in desired response range
    velocity *= (Ts*1000) # Sample every 1 timestep (ms)
    
    # image = np.ones(image.shape) * 255
    
    # image[start[0]:start[0]+target_size, start[1]:start[1]+target_size] = 0
    
    with open(os.path.join(root, 'GroundTruth.txt'), 'w') as f:
        for timestep in range(100):
            # Roll image by specific time
            image = np.roll(image, round(velocity), axis=1)
            
            # Crop to save memory and increase computation speed
            # bg = image[1000:1400,2000:4594].copy()
            bg = image[1000:1000 + pixels_to_keep[0], 2000:2000 + pixels_to_keep[1]].copy()
            
            # bg[start[0]:start[0]+target_size, start[1]+round(velocity)*timestep:start[1]+target_size+round(velocity)*timestep] = 0
            
            # Two targets moving together
            if degree_of_separation is not None:
                separation_distance = round(pixels_per_degree * degree_of_separation)
                bg[start[0]:start[0]+target_size, 
                   start[1]+target_size+round(velocity)*timestep+separation_distance:start[1]+2*target_size+round(velocity)*timestep+separation_distance] = 0
            
            cv2.imwrite(os.path.join(EMD_folder, naming_convention(timestep+1) + '.png'), bg)
            
            x, y = start[0], start[1]+round(velocity)*timestep # start[0]+target_size, start[1]+round(velocity)*timestep
            # Adjust for roll
            if y >= image.shape[1]:
                y -= image.shape[1]
                
            f.write(f'{y},{x},{target_size},{target_size}')
            f.write('\n')
        f.close()
        
    return pixels_to_keep
    
if EMD_folder in root:
    pixels_to_keep = create_EMD_tests()
    pixels_to_keep[1] *= 2
    vf = rf(pixels_to_keep).run()

# Wiederman (2008)
def create_botanic_panorama():
    image = cv2.imread(os.path.join(botanic_folder, 'HDR_Botanic_RGB_lin.tif'))
    
    # Target size definitions
    pixels_per_degree = image.shape[1]/360
    target_size = round(1.6 * pixels_per_degree) # 2.8 degrees is optimal target size
    start = (205, 1225) # 770
    
    image[start[0]:start[0]+target_size, start[1]:start[1]+target_size,:] = 0
        
    # Calculate velocity of image
    velocity = (90/1000) * pixels_per_degree # 130 degrees/second gives optimal latency of 29 ms in desired response range
    velocity *= (Ts*1000) # Sample every 1 timestep (ms)
    
    with open(os.path.join(root, 'GroundTruth.txt'), 'w') as f:
        for timestep in range(100):
            cv2.imwrite(os.path.join(botanic_folder, naming_convention(timestep+1) + '.png'), image)
            
            # Roll image by specific time
            image = np.roll(image, round(velocity), axis=1)
            
            x, y = start[0], start[1]+round(velocity)*timestep # start[0]+target_size, start[1]+round(velocity)*timestep
            # Adjust for roll
            if y >= image.shape[1]:
                y -= image.shape[1]
                
            f.write(f'{y},{x},{target_size},{target_size}')
            f.write('\n')
        f.close()
    
if botanic_folder in root:
    create_botanic_panorama()
    
# Obtain ground truths
if any(x in root for x in [nominal_folder, 'botanic', 'EMD']):
    with open(os.path.join(root, 'GroundTruth.txt')) as groundTruth:
        GT = []
        for line in groundTruth:
            # Save all ground truths as tuples
            tup = eval(line.rstrip())
            start_xy = (tup[0], tup[1])
            end_xy = (tup[0] + tup[2], tup[1] + tup[3])
            GT.append((*start_xy, *end_xy))
    groundTruth.close()
    
# Nordstrom and O'Carroll (2006)
def image_generation(timestep, mode=None, clutter=None):
    
    if not any([mode, clutter]):
        raise TypeError('Missing mode and clutter. Either height or velocity')
        
    images = np.ones((410, 410)) * 255
    
    # Calculate pixels per degree
    pixels_per_degree = images.shape[1] / desired_resolution[1]
    
    # Height tuning
    if mode == 'height':
        target_speed = (50/1000) * pixels_per_degree # 50 degrees per second
        target_size = int(2.1 * pixels_per_degree)
        
    # Velocity tuning
    elif mode == 'velocity':
        target_speed = (args.value/1000) * pixels_per_degree
        target_size = round(0.8 * pixels_per_degree) # 2.2 degrees
        
    if clutter:
        if mode is None:
            target_speed = (300/1000) * pixels_per_degree # 300 degrees per second
            target_size = round(2.1 * pixels_per_degree) # 2.1 degrees
        
        # Images become cluttered
        mean_magnitude = naturalistic_noise.fourier()
        images = mean_magnitude
    
    # Position of target based on target speed
    x_loc = int(target_speed * timestep)
    
    start = (20, 20) # (20, 20)
    
    # Define dark target
    # To ensure rollover
    start_coord = (start[1]+x_loc) - ((start[1]+x_loc) // images.shape[1]) * images.shape[1]
    # start_coord = start[1]+x_loc
    # images[start[0]:start[0]+target_size, start_coord:start_coord+target_size] = 0
    images[start[0]:start[0]+target_size, start_coord:start_coord+int(pixels_per_degree * 8)] = 0
    
    save_dir = os.path.join(os.getcwd(), 'Bagheri', 'Tuning')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, naming_convention(timestep+1) + '.png'), images)
    
if tuning_folder in root:
    for t in range(1500):
        image_generation(t, mode='velocity', clutter=None)
        
def indices_of_max_value(arr):
    # Used to find indices of max value in array (ESTMD_Output)
    # Target in fourth row
    arr_sub = arr[4,:,:]
    return np.unravel_index(np.argmax(arr_sub, axis=None), arr_sub.shape)
    
def tuning_plots(mode, bar=False, velocity_wiederman=False, latency=False):
    if mode == 'height':
        tuning_array_file = 'height_tuning_ESTMD.txt'
        if bar:
            tuning_array_file = 'tuning_plots_height_bar.txt'
        label = 'Target Height [$\degree$]'
    elif mode == 'velocity':
        tuning_array_file = 'velocity_tuning_ESTMD.txt'
        label = 'Target Velocity [$\degree$/s]'
    elif mode == 'FDSR':
        tuning_array_file = 'velocity_tuning_ESTMD.txt'
        label = 'Target Velocity [$\degree$/s]'
    else:
        raise ValueError('Incorrect mode. Check spelling of mode in call')
    
    with open(tuning_array_file, "r") as file:
        tuning_array = []
        # Read file
        for line in file:
            values = eval(line.rstrip())
            tuning_array.append(values)
    
    tuning_array = np.asarray(tuning_array).T
    tuning_array[1,:] = tuning_array[1,:] / np.max(tuning_array[1,:])
    
    import pandas as pd
    
    def read_csv(dataset_csv):
        df = pd.read_csv(dataset_csv)
        wiederman = df.iloc[:,[0,1]].to_numpy()[1:].astype(float)
        wiederman = wiederman[~np.isnan(wiederman).any(axis=1)]
        
        physiology = df.iloc[:,[2,3]].to_numpy()[1:].astype(float)
    
        return wiederman, physiology
    
    from helper_functions import set_size
    
    fig, axes = plt.subplots(figsize=(7,6), dpi=500)
    
    if bar:
        wiederman, physiology = read_csv('C:/Users/ag803/STMD/STMD paper assets/wpd_datasets_size.csv')
        estmd = axes.plot(tuning_array[0,:], tuning_array[1,:], '-o', label='ESTMD model reproduction', markersize=6, color='crimson')
        wiederman_model = axes.plot(wiederman[:,0], wiederman[:,1] / np.max(wiederman[:,1]), '-^', label='Wiederman et al. (2008) model', markersize=6, color='black')
        ephys = axes.errorbar(physiology[::3,0], physiology[::3,1]  / np.max(physiology[::3,1]), yerr=np.abs(np.c_[physiology[2::3,1], physiology[1::3,1]].T - physiology[::3,1]), 
                             label='Physiology STMD', marker="s", markersize=6, color='green')
        axs = estmd + wiederman_model
        labs1 = [l.get_label() for l in axs]
    elif mode == 'velocity' and velocity_wiederman:
        wiederman, physiology = read_csv('C:/Users/ag803/STMD/STMD paper assets/wpd_datasets_velocity.csv')
        estmd = axes.plot(tuning_array[0,:], tuning_array[1,:], '-o', label='ESTMD model reproduction', markersize=6, color='crimson')
        wiederman_model = axes.plot(wiederman[:,0], wiederman[:,1] / np.max(wiederman[:,1]), '-^', label='Wiederman et al. (2008) model', markersize=6, color='black')
        ephys = axes.errorbar(physiology[::3,0], physiology[::3,1] / np.max(physiology[::3,1]), yerr=np.abs(np.c_[physiology[2::3,1], physiology[1::3,1]].T - physiology[::3,1]), 
                              label='Physiology STMD', marker="s", markersize=6, color='green')
        axs = estmd + wiederman_model
        labs1 = [l.get_label() for l in axs]
    elif mode == 'FDSR':
        estmd = axes.plot(tuning_array[0,:], tuning_array[1,:], '-', label='$\\tau_{FAST}=10 ms$, $\\tau_{SLOW}=100 ms$', markersize=6, color='crimson')
        df = pd.read_csv('STMD paper assets/FDSR_tuning.csv')
        FDSR_arr = df.iloc[:,:].to_numpy()[1:].astype(float)

        tau_fast = axes.plot(FDSR_arr[:,0], FDSR_arr[:,1] / np.max(FDSR_arr[:,1]), '-', 
                             label='$\\tau_{FAST}=5 ms$, $\\tau_{SLOW}=100 ms$', markersize=6, color='blue')
        tau_slow = axes.plot(FDSR_arr[:,0], FDSR_arr[:,2] / np.max(FDSR_arr[:,2]), '-', 
                             label='$\\tau_{FAST}=10 ms$, $\\tau_{SLOW}=50 ms$', markersize=6, color='green')
        rectangular_profile = axes.plot(FDSR_arr[:,0], FDSR_arr[:,3] / np.max(FDSR_arr[:,3]), '-', 
                                        label='$\\tau_{FAST}=10 ms$, $\\tau_{SLOW}=100 ms$', markersize=6, color='orange')
        axs = estmd + tau_fast + tau_slow + rectangular_profile
        labs1 = [l.get_label() for l in axs]
    elif latency:
        axes.plot(tuning_array[0,:], tuning_array[2,:], '-r^', markersize=4)
        axes.axhline(y=20, color='black', linestyle='dashed')
        axes.axhline(y=40, color='black', linestyle='dashed')
    else:
        axes.plot(tuning_array[0,:], tuning_array[1,:], '-o', markersize=6)
        if mode == 'height':
            axes.axvline(x=1.6, color='black', linestyle='dashed')
    axes.set_xlim([0, None])
    
    # if mode == 'velocity':
    #     col = np.argwhere(tuning_array == 300)[0][1]
    #     axes.plot(tuning_array[0,col], tuning_array[1,col], 'k*', markersize=10)
    #     ax2.plot(tuning_array[0,col], tuning_array[2,col], 'k*', markersize=10)
    if bar or (mode == 'velocity' and velocity_wiederman):
        if bar:
            axes.set_xlim([0.1, 100]) # To match Wiederman et al. (2008) paper
        else:
            axes.set_xlim([1, 1000])
        axes.set_xscale('log')
        axes.legend(axs + [ephys[0]], labs1 + [ephys.get_label()])
    elif mode == 'velocity':
        if (mode == 'velocity' and not velocity_wiederman) or latency:
            axes.set_xscale('log')
            axes.set_xlim([1, 1000])
    elif mode == 'FDSR':
        axes_sqs = []
        for i in range(len(axs)):
            if i < 3:
                axes_sqs.append(plt.Line2D([], [], marker="s", markersize=5, linewidth=0, color=axs[i]._color))
            else:
                axes_sqs.append(plt.Line2D([], [], marker="s", markersize=5, linewidth=6, color=axs[i]._color))
        axes.legend(axes_sqs, labs1)
        axes.set_xlim([1, 1000])
        axes.set_xscale('log')
    axes.set_ylim([0, None])
    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:g}'.format(x)))
    axes.set_xlabel(label)
    if latency:
        axes.set_ylabel('Latency (ms)')
    else:
        axes.set_ylabel('ESTMD Response (normalised)')
    
# tuning_plots('height', bar=True)

def kernel_2D_results():
    headings = []
    values = []
    kernel_vals = []
    
    with open('kernel_2D_vs_1D_comparison.txt', 'r') as f:
        for line in f:
            if ',' not in line:
                headings.append(line.rstrip('\n'))
                if not len(values) == 0:
                    kernel_vals.append(np.stack(values))
                    values = []
            else:
                values.append(np.array(eval(line)))
        kernel_vals.append(np.stack(values))
                
    all_vals = np.c_[[kernel_vals[i][:,1] for i in range(len(kernel_vals))]]
    vals = np.c_[kernel_vals[0][:,0], all_vals.T / np.max(all_vals)]
    
    fig, axes = plt.subplots(dpi=500)
    for i in range(vals.shape[1]-1):
        plt.plot(vals[:,0], vals[:,i+1], label=headings[i])
    axes.set_xlabel('Degree of separation (degrees)')
    axes.set_ylabel('ESTMD Magnitude (normalised)')
    axes.set_xticks(np.arange(0, np.max(vals[:,0]), 2))
    axes.legend()
    
# kernel_2D_results()

def Bagheri_load_results():
    # Percentage time in bounding box
    results_row = scipy.io.loadmat(os.path.join(nominal_folder, 'results_Facil_on', 'TargetLocationRow.mat'))['TargetLocationRow']
    results_col = scipy.io.loadmat(os.path.join(nominal_folder, 'results_Facil_on', 'TargetLocationCol.mat'))['TargetLocationCol']
    
    results_row_off = scipy.io.loadmat(os.path.join(nominal_folder, 'results_Facil_off', 'TargetLocationRow.mat'))['TargetLocationRow']
    results_col_off = scipy.io.loadmat(os.path.join(nominal_folder, 'results_Facil_off', 'TargetLocationCol.mat'))['TargetLocationCol']
    
    return np.hstack((results_row, results_col)), np.hstack((results_row_off, results_col_off))

def clamp(n, smallest, largest):
    return [max(smallest, min(x, largest)) for x in n]

def metric(bbox_ds, target_loc, success_MATLAB_counter, success_python_counter, delay):
    for timestep, bbox in enumerate(bbox_ds[:-delay]):
        # Ordinate
        x_left, x_right = clamp([bbox[0], bbox[2]], 0, ds_size[1])
        # Abscissa
        y_left, y_right = clamp([bbox[1], bbox[3]], 0, ds_size[0])
        
        # Coordinates in bounding boxes
        x, y = np.mgrid[y_left:y_right, x_left:x_right]
        xy = np.vstack((x.flat, y.flat)).T
        
        # Check if target in frame
        if xy.size > 0:
            # Python success (This implementation)
            pix_in_bbox = ESTMD_Output[xy[:,0],xy[:,1],timestep+delay]
            if not all([v == 0 for v in pix_in_bbox]):
                if np.max(pix_in_bbox)/np.max(ESTMD_Output[:,:,timestep+delay]) >= 0.99:
                    success_python_counter.append(success_python_counter[-1] + 1)
                else:
                    success_python_counter.append(success_python_counter[-1])
            else:
                success_python_counter.append(success_python_counter[-1])
            
            # MATLAB success (Bagheri implementation)
            for i, tl in enumerate(target_loc):
                tl_ = tl[timestep+delay,:]
                if any(np.equal(tl_, xy).all(axis=1)):
                    success_MATLAB_counter[i].append(success_MATLAB_counter[i][-1] + 1)
                else:
                    success_MATLAB_counter[i].append(success_MATLAB_counter[i][-1])
    
    return success_MATLAB_counter, success_python_counter

# Widefield stimuli to check RTC as in Wiederman paper (2008)
def widefield_stimuli():
    PULSE_WIDTH = 5 # Pulse width
    INTERVAL = 10
    TIMESTEPS = (INTERVAL + PULSE_WIDTH) * 20 # Timestep to see adaptation
        
    images = np.ones((480, 640, TIMESTEPS)) * 128
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
    mean_magnitude = naturalistic_noise.fourier()
    
    for i in range(images.shape[-1]):
        images[:,:,i] = (images[:,:,i] + mean_magnitude) / 2
        cv2.imwrite(os.path.join(save_images_root, naming_convention(i+1) + '.png'), images[:,:,i])
                    
    return images

# if image_folder in root:
#     images = widefield_stimuli()
    
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

def folder_name(save_dir_type):
    folder_num = root.split('/')[-1]
    return '_'.join([save_dir_type, folder_num])

def ESTMD_delay(ESTMD_var, bbox_ds, delay):
    ESTMD = ESTMD_var[:,:,delay:]
    for i in range(ESTMD.shape[2]):
        snap = ESTMD[:,:,i]
        # bbox_decision = lambda root: bbox_ds[i] if nominal_folder in root else None
        bounding_box(snap, bbox_ds[i], i)
        
def bounding_box(image, bbox, t, upscale=None):
    fig, axes = plt.subplots(figsize=(48,36))
    axes.imshow(image, interpolation=None, cmap='gray')
    
    if bbox is not None:
        x, y = bbox[:2]
        tup_subtract = tuple(map(lambda i, j: i - j, bbox[-2:], bbox[:2]))
        width, height = tup_subtract
    
        rect = plt.Rectangle((x-1.5, y+0.5), width-1, height-1, fill=False, color="limegreen", linewidth=1)
        axes.add_patch(rect)
    hide_axes(axes)
    
    number = naming_convention(t+1)
    
    if upscale is not None:
        plt.text(0.2, 0.9, 'ESTMD Bagheri', fontsize=12, transform=plt.gcf().transFigure)
        ESTMD_folder = 'MATLAB_Bagheri'
    else:
        ESTMD_folder = 'vid_ESTMD'
        
    save_fig(number, folder_name(ESTMD_folder))

def upscale_MATLAB_figs(delay):    
    vid_folder = 'C:/Users/ag803/STMD/4496768/STNS3/28/ESTMD_Facil'
    files = [f'ESTMD_{i+delay+1}.png' for i in range(len(files)-delay)]
    for t, filename in enumerate(files):
        img = cv2.imread(os.path.join(vid_folder, filename))
        bbox = GT[t]
        bbox = tuple(map(lambda x: round(x/pixel2PR), bbox))
        bounding_box(img.copy(), bbox, t, upscale=True)

# upscale_MATLAB_figs(delay=10)

"""
Initialisations
"""

def initialisations(degrees_in_image):
    image_size = cv2.imread(os.path.join(root, files[0])).shape[:-1]
    pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
    pixel2PR = int(pixels_per_degree)
    ds_size = (tuple(int(np.ceil(x/pixel2PR)) for x in image_size))
    
    return image_size, ds_size

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

def matlab_style_conv2(x, y, mode='same', **kwargs):
    """
    should give the same result as MATLAB's
    conv2(x, y, mode='same') with padding
    """
    pad_width = kwargs.pop('pad_width', 0)
    # Add padding
    padded = np.pad(x, pad_width=pad_width, mode='reflect')
    convolved = np.rot90(scipy.signal.convolve2d(np.rot90(padded, 2), np.rot90(y, 2), mode=mode), 2)
    
    # Remove padding
    return convolved[pad_width:-pad_width, pad_width:-pad_width]
    
degrees_in_image = desired_resolution[1]
image_size, ds_size = initialisations(degrees_in_image)

# sf = np.zeros((*image_size, len(files)))
on_f = np.zeros((*ds_size, len(files)), dtype=np.float16)
off_f = np.zeros_like(on_f)
fdsr_on = np.zeros_like(on_f)
fdsr_off = np.zeros_like(on_f)
# delayed_on = np.zeros_like(on_f)
# delayed_off = np.zeros_like(on_f)
ESTMD_Output = np.zeros_like(on_f)
RTC_Output = np.zeros_like(on_f)

bbox_ds = []

delay = 10
STMD_all = []
LPTC_all = []
wSTMD_all = []

spontaneous = []
LPTC_HR = []

"""
Simulation of ESTMD
"""
for t, file in enumerate(files):
    image = cv2.imread(os.path.join(root, file))
    
    if any(x in root for x in [nominal_folder, 'botanic', 'EMD']):
        # Bounding box params
        bbox = GT[t]
        
        # Image with bounding box
        if t < len(files) - delay:
            image_save = cv2.rectangle(image.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            number = naming_convention(t+1)
            rf = True if 'EMD' in root else False
            # visualise([image_save[165:265, 1124:1398]], folder_name('vid_images'), title=[number], rf=rf)
    
    # Extract green channel from BGR
    green = image[:,:,1]
    
    image_size = green.shape
    pixels_per_degree = image_size[1]/degrees_in_image # horizontal pixels in output / horizontal degrees (97.84)
    pixel2PR = int(pixels_per_degree) # ratio of pixels to photoreceptors in the bio-mimetic model (1 deg spatial sampling... )
    
    sigma = 1.4 / (2 * np.sqrt(2 * np.log(2)))
    sigma_pixel = sigma * pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree
    kernel_size = int(6 * sigma_pixel - 1)
    pad_width = int(kernel_size / 2)
    
    """Downsampling receptive fields"""
    if EMD_folder in root:
        Downsampledvf = cv2.resize(vf, np.flip(ds_size), interpolation=cv2.INTER_NEAREST)
    
    # Spatial filtered through LPF1
    # Gaussian kernel
    H = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma_pixel)
    sf = matlab_style_conv2(green, H, mode='same', pad_width=pad_width)
    
    # Scipy's way - Interpolation is too good for puny humans
    # sf = gaussian_filter(green, radius=kernel_size, sigma=sigma_pixel)
    
    # Downsampled green channel
    # Subsampling vs Downsampling
    # XXX: Downsampling makes more sense. Check how much the change affects results after everything is working
    # DownsampledGreen = sf[::pixel2PR, ::pixel2PR]
    DownsampledGreen = cv2.resize(sf, np.flip(ds_size), interpolation=cv2.INTER_NEAREST)
    
    # visualise([DownsampledGreen], folder_name('STNS3_downsampled'), title=[number])
    
    # Downsampled bbox
    if any(x in root for x in [nominal_folder, 'botanic', 'EMD']):
        bbox = tuple(map(lambda x: round(x/pixel2PR), bbox))
        bbox_ds.append(bbox)
    
    try:
        dbuffer1
    except NameError:
        dbuffer1 = np.zeros((*DownsampledGreen.shape, len(photo_z["b"])))
    
    # Photoreceptor output after temporal band-pass filtering
    PhotoreceptorOut, dbuffer1 = IIR_Filter(photo_z["b"], photo_z["a"], DownsampledGreen/255, dbuffer1)
    # visualise([PhotoreceptorOut], folder_name('STNS3_photoreceptor'), title=[number])
    
    # LMC output after spatial high pass filtering
    LMC_Out = matlab_style_conv2(PhotoreceptorOut, CSA_KERNEL, mode='same', pad_width=pad_width)
    
    # visualise([LMC_Out], folder_name('STNS3_LMC'), title=[number])
    
    # Half-wave rectification
    # Clamp the high pass filtered data to separate the on and off channels
    on_f[:,:,t] = np.maximum(LMC_Out, 0.0)
    off_f[:,:,t] = -np.minimum(LMC_Out, 0.0)
    
    # visualise([off_f[:,:,t]], folder_name('STNS3_off'), title=[number])
    
    if t > 0:
        # FDSR Implementation
        k_on = np.where((on_f[:,:,t] - on_f[:,:,t-1]) > 0.01, FDSR_K_FAST_ON, FDSR_K_SLOW)
        k_off = np.where((off_f[:,:,t] - off_f[:,:,t-1]) > 0.01, FDSR_K_FAST_OFF, FDSR_K_SLOW)
        # Doesn't match Weiderman paper but is the only thing that makes sense
        
        # Apply low-pass filters to on and off channels
        fdsr_on[:,:,t] = ((1.0 - k_on) * on_f[:,:,t]) + (k_on * fdsr_on[:,:,t-1])
        fdsr_off[:,:,t] = ((1.0 - k_off) * off_f[:,:,t]) + (k_off * fdsr_off[:,:,t-1])
        
        # Subtract FDSR from half-wave rectified
        a_on = (on_f[:,:,t] - fdsr_on[:,:,t]).clip(min=0)
        a_off = (off_f[:,:,t] - fdsr_off[:,:,t]).clip(min=0)
        
        # visualise([a_off], folder_name('botanic_off_FDSR'), title=[number])
        
        # Inhibition is implemented as spatial filter
        # Not true if this is a chemical synapse as this would require delays
        # Half-wave rectification added
        on_filtered = matlab_style_conv2(a_on, INHIB_KERNEL, mode='same', pad_width=pad_width).clip(min=0)
        off_filtered = matlab_style_conv2(a_off, INHIB_KERNEL, mode='same', pad_width=pad_width).clip(min=0)
        
        # visualise([on_filtered], folder_name('STNS3_off_filtered_images_2D'), title=[number])
        # visualise([off_filtered], folder_name('botanic_off_filtered_images_2D'), title=[number])
        
        try:
            ONbuffer
            OFFbuffer
        except NameError:
            ONbuffer = np.zeros((*on_filtered.shape, len(LPF_5["b"])))
            OFFbuffer = np.zeros((*off_filtered.shape, len(LPF_5["b"])))
        
        # Delayed channels using z-transform
        On_Delayed_Output, ONbuffer = IIR_Filter(LPF_5["b"], LPF_5["a"], on_filtered, ONbuffer)
        Off_Delayed_Output, OFFbuffer = IIR_Filter(LPF_5["b"], LPF_5["a"], off_filtered, OFFbuffer)
        
        # Delayed channels for LPF5
        # delayed_on[:,:,t] = ((1.0 - LPF5_K) * on_filtered) + (LPF5_K * delayed_on[:,:,t-1])
        # delayed_off[:,:,t] = ((1.0 - LPF5_K) * off_filtered) + (LPF5_K * delayed_off[:,:,t-1])
        
        # Correlation between channels
        Correlate_ON_OFF = on_filtered * Off_Delayed_Output # delayed_off[:,:,t]
        Correlate_OFF_ON = off_filtered * On_Delayed_Output # delayed_on[:,:,t]
        Correlate_OFF_ON = np.zeros_like(Correlate_OFF_ON)
        
        RTC_Output[:,:,t] = (Correlate_ON_OFF + Correlate_OFF_ON) * 6
        ESTMD_Output[:,:,t] = (RTC_Output[:,:,t]).clip(min=0)
        # ESTMD_Output[:,:,t] = np.tanh(ESTMD_Output[:,:,t])
        
#     """
#     ESTMD -> EMD -- Directionally selective STMD
#     """
#     # TODO: Just moving right for the meantime
#     try:
#         EHR_buffer_right
#     except NameError:
#         EHR_buffer_right = np.zeros((*ESTMD_Output.shape[:-1], len(LPF_HR["b"])))
    
#     # Delayed channels using z-transform for HR
#     EHR_Delayed_Output_right, EHR_buffer_right = IIR_Filter(LPF_HR["b"], LPF_HR["a"], ESTMD_Output[:,:,t].copy(), EHR_buffer_right)
    
#     # Correlate delayed channels
#     EHR_right = (EHR_Delayed_Output_right[:,:-1] * ESTMD_Output[:,1:,t]) - (ESTMD_Output[:,:-1,t] * EHR_Delayed_Output_right[:,1:])
    
#     # Half-wave rectification
#     if EMD_folder in root:
#         EHR_R = np.maximum(EHR_right, 0.0) * Downsampledvf[:,:-1,1]
#         EHR_L = -np.minimum(EHR_right, 0.0) * Downsampledvf[:,:-1,1]
    
#     if t < len(files) - delay:
#         number = naming_convention(t+1)
#         # visualise([EHR_R], folder_name('vid_dSTMD_R'), title=[number])
#         # visualise([EHR_L], folder_name('vid_dSTMD_L'), title=[number])
    
#     # Spatially pool d-STMD
#     if EMD_folder in root:
#         STMD_sp = np.array([np.sum(EHR_R), np.sum(EHR_L)])
#         STMD_all.append(STMD_sp)
#         STMD_sp_sum = np.sum([np.sum(EHR_R), np.sum(EHR_L)])
    
#     #TODO: Plots of cardinal direction STMDs, sum of STMDs compared with EMDS
    
#     """
#     Wide-field directionally selective EMD
#     """
    
#     # HR-Correlator -- EMD - right preferred direction
#     try:
#         ON_HR_buffer_right
#         OFF_HR_buffer_right
#     except NameError:
#         ON_HR_buffer_right = np.zeros((*on_f.shape[:-1], len(LPF_HR["b"])))
#         OFF_HR_buffer_right = np.zeros((*off_f.shape[:-1], len(LPF_HR["b"])))
    
#     # Delayed channels using z-transform for HR
#     On_HR_Delayed_Output_right, ON_HR_buffer_right = IIR_Filter(LPF_HR["b"], LPF_HR["a"], on_f[:,:,t].copy(), ON_HR_buffer_right)
#     Off_HR_Delayed_Output_right, OFF_HR_buffer_right = IIR_Filter(LPF_HR["b"], LPF_HR["a"], off_f[:,:,t].copy(), OFF_HR_buffer_right)
    
#     # Correlate delayed channels
#     on_HR_right = (On_HR_Delayed_Output_right[:,:-1] * on_f[:,1:,t]) - (on_f[:,:-1,t] * On_HR_Delayed_Output_right[:,1:])
#     off_HR_right = (Off_HR_Delayed_Output_right[:,:-1] * off_f[:,1:,t]) - (off_f[:,:-1,t] * Off_HR_Delayed_Output_right[:,1:])
    
#     # EMD_Output_right = (on_HR_right + off_HR_right) * 6
#     EMD_Output_right = off_HR_right * 6
#     EMD_Output_right[np.abs(EMD_Output_right) < 0.01] = 0
#     EMD_Output_right = np.tanh(EMD_Output_right)
    
#     # HR-Correlator -- EMD - down preferred direction
#     try:
#         ON_HR_buffer_down
#         OFF_HR_buffer_down
#     except NameError:
#         ON_HR_buffer_down = np.zeros((*on_f.shape[:-1], len(LPF_HR["b"])))
#         OFF_HR_buffer_down = np.zeros((*off_f.shape[:-1], len(LPF_HR["b"])))
    
#     # Delayed channels using z-transform for HR
#     On_HR_Delayed_Output_down, ON_HR_buffer_down = IIR_Filter(LPF_HR["b"], LPF_HR["a"], on_f[:,:,t].copy(), ON_HR_buffer_down)
#     Off_HR_Delayed_Output_down, OFF_HR_buffer_down = IIR_Filter(LPF_HR["b"], LPF_HR["a"], off_f[:,:,t].copy(), OFF_HR_buffer_down)
    
#     # Correlate delayed channels
#     on_HR_down = (On_HR_Delayed_Output_down[:-1,:] * on_f[1:,:,t]) - (on_f[:-1,:,t] * On_HR_Delayed_Output_down[1:,:])
#     off_HR_down = (Off_HR_Delayed_Output_down[:-1,:] * off_f[1:,:,t]) - (off_f[:-1,:,t] * Off_HR_Delayed_Output_down[1:,:])
    
#     # EMD_Output_down = (on_HR_down + off_HR_down) * 6
#     EMD_Output_down = off_HR_down * 6
#     EMD_Output_down[np.abs(EMD_Output_down) < 0.01] = 0
#     EMD_Output_down = np.tanh(EMD_Output_down)
    
#     EMD_Output_R = np.maximum(EMD_Output_right, 0.0)
#     EMD_Output_L = -np.minimum(EMD_Output_right, 0.0)
    
#     # Add Poisson baseline
#     LPTC_spontaneous = np.random.poisson(20)
#     LPTC_baseline = LPTC_spontaneous + np.sum(EMD_Output_right) # right is preferred direction of LPTC
#     LPTC_HR.append(LPTC_baseline)
#     spontaneous.append(LPTC_spontaneous)
    
#     # Spatially pool EMD
#     EMD_sp = np.array([np.sum(EMD_Output_R), np.sum(EMD_Output_L)])
#     LPTC_all.append(EMD_sp)
    
#     """
#     Wide-field directionally selective STMD
#     """
    
#     if EMD_folder in root:
#         wEHR_R = np.maximum(EHR_right, 0.0) * Downsampledvf[:,:-1,2]
#         wEHR_L = -np.minimum(EHR_right, 0.0) * Downsampledvf[:,:-1,2]
    
#     if t < len(files) - delay:
#         number = naming_convention(t+1)
#         # visualise([wEHR_R], folder_name('vid_wSTMD_R'), title=[number])
#         # visualise([wEHR_L], folder_name('vid_wSTMD_L'), title=[number])
    
#     # Spatially pool d-STMD
#     if EMD_folder in root:
#         wSTMD_sp = np.array([np.sum(wEHR_R), np.sum(wEHR_L)])
#         wSTMD_all.append(wSTMD_sp)
#         wSTMD_sp_sum = np.sum([np.sum(wEHR_R), np.sum(wEHR_L)])

# if EMD_folder in root:    
#     STMD_all = np.stack((STMD_all))
#     LPTC_all = np.stack((LPTC_all))
#     wSTMD_all = np.stack((wSTMD_all))

# # Rightward motion first
# if EMD_folder in root:
#     print(f'STMD:{np.sum(STMD_all[:,0])},{np.sum(STMD_all[:,1])};'
#           f'LPTC:{np.sum(LPTC_all[:,0])},{np.sum(LPTC_all[:,1])};'
#           f'wSTMD:{np.sum(wSTMD_all[:,0])},{np.sum(wSTMD_all[:,1])};'
#           f'LPTC_baseline:{np.sum(LPTC_baseline)}')
    
    # print(f'{np.sum(LPTC_baseline)},{np.mean(spontaneous)}')

# print(f'{t},{np.sum(EMD_Output_right)},{np.sum(EMD_Output_down)}')

# print(f'{t},{np.sum(ESTMD_Output[:,:,-1])}')
# print(f'{t},{np.sum(EMD_Output_right)},{np.sum(EMD_Output_down)}')
    
results_on, results_off = Bagheri_load_results()

success_MATLAB_counter = [[0], [0]]
success_python_counter = [0]

success_MATLAB_counter, success_python_counter = metric(bbox_ds, [results_on, results_off], 
                                                        success_MATLAB_counter, success_python_counter, delay=delay)

total_frames = len(success_python_counter)

if nominal_folder in root:
    print(nominal_folder, f'{success_MATLAB_counter[0][-1]/total_frames*100},'
          f'{success_MATLAB_counter[1][-1]/total_frames*100},'
          f'{success_python_counter[-1]/total_frames*100}')

if any(x in root for x in [nominal_folder, 'botanic']):
    ESTMD_delay(ESTMD_Output, bbox_ds, delay=delay)

def success_plots(success_MATLAB_counter, success_python_counter):
    fig, axes = plt.subplots(figsize=(12,9))
    MATLAB_titles = ['Facilitation ON - 200 ms', 'Facilitation OFF']
    for m, counter in enumerate(success_MATLAB_counter):
        axes.plot(np.asarray(counter)/total_frames * 100, label=MATLAB_titles[m])
    axes.plot(np.asarray(success_python_counter)/total_frames * 100, label='Python')
    axes.minorticks_on()
    axes.yaxis.set_tick_params(which='minor', left=False)
    axes.set_xlabel('Frames')
    axes.set_ylabel('Success rate [%]')
    axes.legend()
    
# success_plots(success_MATLAB_counter, success_python_counter)

def latency(arr):
    activations = []
    for t in range(arr.shape[-1]):
        activations.append(np.sum((arr[...,t] - 0.01).clip(min=0)))
    activations_adjusted = np.flatnonzero(activations)
    if len(activations_adjusted) > 0:
        return activations_adjusted[0]
    else:
        return None

# For quick shell script
if tuning_folder in root:
    # print(f'{args.value},{ESTMD_Output[(2,)+indices_of_max_value(ESTMD_Output)]},{indices_of_max_value(ESTMD_Output)[-1]}')
    # print(f'{args.value},{np.sum(ESTMD_Output)},{latency(ESTMD_Output)}')
    print(f'{np.sum(ESTMD_Output)}')

# print(f'{args.value},{np.sum(ESTMD_Output[205 // pixel2PR:(221 // pixel2PR) + 1,...])}')
# print(f'{args.value},{np.sum(ESTMD_Output)}')

def _RTC(vars, **kwargs):
    # Initialise ragged nested list
    var_vals = [[[] for i in range(len(j))] for j in vars]
    for v, var in enumerate(vars):
        for i, var_t in enumerate(var):
            for j in range(var_t.shape[-1]):
                # All values in photoreceptor array should be the same
                frame = var_t[:,:,j]
                coord_xy = (np.array(frame.shape)/2).astype(int)
                # TODO: Find out what is going on in pixel
                var_vals[v][i].append(frame[coord_xy[0], coord_xy[1]])
            
        fig, ax = plt.subplots(figsize=(12,9))
        for i, val in enumerate(var_vals[v]):
            ax.plot(val)
        ax.set_xlabel('Time [msec]')
        if len(var) > 1:
            ax.legend(labels = ['On', 'Off'])
        ax.get_yaxis().set_visible(False)
        
        save_fig(kwargs['title'][v], 'RTC_images')
        
    return var_vals

# if image_folder in root:
#     _RTC([[sf], [PhotoreceptorOut], [LMC_Out], [fdsr_on, fdsr_off], [ESTMD_Output], [RTC_Output], [on_filtered, delayed_off], 
#           [delayed_on, off_filtered], [on_f, off_f], [a_on, a_off]], 
#           title=['sf', 'PhotoreceptorOut', 'LMC_Out', 'FDSR', 'ESTMD', 'RTC', 'On_D_OFF', 'Off_D_ON', 'HWR1', 'After FDSR'])

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
        
def image2Video(layer_name):
    img_array = []
    vid_folder = os.path.join(os.getcwd(), 'Bagheri', folder_name('_'.join(['vid', layer_name])))
    video_name = folder_name(layer_name)
    files = [file for file in os.listdir(vid_folder) if file.endswith('.png')]
    for filename in files:
        img = cv2.imread(os.path.join(vid_folder, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        
    out = cv2.VideoWriter(video_name + '_updated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    # ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4
    """For adding coloured padding in vstack
    ffmpeg -i output_EMD.mp4 -i dSTMD_output.mp4 -filter_complex 
    "[1]pad=iw:ih+5:0:5:color=white[v1];[0][v1]vstack=inputs=2" dSTMD_output_all.mp4"""
    # os.system(f"ffmpeg -i {folder_name('images') + '.mp4'} \
    #           -i {folder_name('ESTMD') + '.mp4'} -filter_complex hstack {folder_name('output') + '.mp4'}")

# continuous(photo_z["b"], photo_z["a"])
# __plots()
# image2Video('images')
# image2Video('ESTMD')