"""
Adapted from Jamie's code available at https://github.com/neworderofjamie/hover
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#%%
"""
Params
"""
def visualise(vars, labels=None, **kwargs):
    height, width = vars[0].shape[1:]
    fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(15,10))
    for i in range(height):
        for j in range(width):
            actors = [axes[i,j].plot(v[:,i,j])[0] for v in vars]
            
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
    
    if labels is not None:
        fig.legend(actors, labels, loc="lower center", ncol=len(labels))
        
    save_fig(kwargs['title'])

# Helper function to hide axes ticks etc on axis
def hide_axes(axis):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

# Helper function to save out images from different stages    
def save_fig(title):
    save_out = os.path.join(os.getcwd(), 'figs')
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title))
    plt.close()

# Single panel visualisation
def single_vis(vars, labels=None, **kwargs):
    fig, axes = plt.subplots(figsize=(15,10))
    row, col = kwargs['fig_index']
    actors = [axes.plot(v[:,row,col])[0] for v in vars]
    
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
    if labels is not None:
        fig.legend(actors, labels, loc="lower center", ncol=len(labels), fontsize=25)
        
    save_fig(kwargs['title'])
    
# Single panel visualisation for imshow
def imshow_single_vis(vars, vmin=None, vmax=None, labels=None, **kwargs):        
    height = len(vars)
    fig, axes = plt.subplots(height, 1, sharex=True, sharey=True, figsize=(15,10))
    for i in range(height):
        try:
            axes[i].imshow(vars[i], cmap="gray", vmin=vmin, vmax=vmax)
            
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        except:
            axes.imshow(vars[i], cmap="gray", vmin=vmin, vmax=vmax)
            
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
    
    fig.tight_layout(pad=0)
    save_fig(kwargs['title'])
    
# Resolution of ommatidia array and number of timesteps to simulate
WIDTH = 16
HEIGHT = 16
TIMESTEPS = 16

#%%
"""
Stimulus Generation
"""
# Size of stimuli and its y coordinate
STIM_WIDTH = 2
STIM_HEIGHT = 2
STIM_Y = 8

obj_colour = 'black'
fig_index = (STIM_Y, 4)

t1 = time.perf_counter()

# Generate TIMESTEPS long 'video' of stimuli moving across frame 
images = np.ones((TIMESTEPS, HEIGHT, WIDTH))

for i in range(TIMESTEPS):
    images[i,STIM_Y:STIM_Y+STIM_HEIGHT,i:i+STIM_WIDTH] = 0.0
    
if obj_colour == 'black': # black-on-white
    pass
else:
    images = np.ones(images.shape) - images
    
# Visualize stimuli
fig, axes = plt.subplots(1,TIMESTEPS, sharey=True, figsize=(15,5))
for t, a in enumerate(axes):
    a.imshow(images[t,:,:], cmap="gray")
    hide_axes(a)
fig.tight_layout(pad=0)
save_fig('stimuli.png')
imshow_single_vis([images[fig_index[1],:,:]], title='stimuli_single.png')

#%%
"""
High-pass filters, HP3
"""

# First layer of the model is a high-pass filter which has 
# the effect of turning frames into DVS style on and off events

# Used a low-pass filter with spectral reversal

# LPF_TAU = 0.4

# # EMA time constant
# LPF_K = np.exp(-1 / LPF_TAU)

# lpf = np.ones((TIMESTEPS, HEIGHT, WIDTH))
# f = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
# for t in range(1, TIMESTEPS):
#     lpf[t] = (LPF_K * lpf[t - 1]) + ((1.0 - LPF_K) * images[t])
#     f[t] = (images[t] - lpf[t])

HPF_TAU = 40

# EMA time constant
HPF_K = np.exp(-1 / HPF_TAU)

hpf = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
f = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
for t in range(1, TIMESTEPS):
    hpf[t] = ((1.0 - HPF_K) * hpf[t - 1]) + ((1.0 - HPF_K) * (images[t] - images[t - 1]))
    f[t] = hpf[t]

visualise([f], title='HP3')
single_vis([f], fig_index=fig_index, title='HP3_single')

#%%
"""
Half-wave rectification
"""

# Clamp the high pass filtered data to separate the on and off channels
on_f = np.maximum(f, 0.0)
off_f = -np.minimum(f, 0.0)

visualise([on_f, off_f], ["On", "Off"], title='HW-R after HPF3')
single_vis([on_f, off_f], ["On", "Off"], fig_index=fig_index, title='HW-R after HPF3_single')

#%%
"""
Calculate low-pass filtering
"""

# Convert time constant of low pass filter into a decay to use in a simple low-pass filter
INHIB_LPF_TAU = 2.0
INHIB_LPF_K = np.exp(-1 / INHIB_LPF_TAU)

# Apply low-pass filter to on and off channel
on_inhib_lpf = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
off_inhib_lpf = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
for t in range(1, TIMESTEPS):
    on_inhib_lpf[t] = (INHIB_LPF_K * on_inhib_lpf[t - 1]) + ((1.0 - INHIB_LPF_K) * on_f[t])
    off_inhib_lpf[t] = (INHIB_LPF_K * off_inhib_lpf[t - 1]) + ((1.0 - INHIB_LPF_K) * off_f[t])

visualise([on_inhib_lpf, off_inhib_lpf], ["On LPF", "Off LPF"], title='LP4')
single_vis([on_inhib_lpf, off_inhib_lpf], ["On LPF", "Off LPF"], fig_index=fig_index, title='LP4_single')

#%%
"""
Calculate inhibition
"""

# Define 1D kernel
INHIB_KERNEL = 0.2 * np.asarray([2.0, 1.0, 0.0, 1.0, 2.0])

# Convolve the 1D kernel with low-pass filtered input every timestep
on_conv_inhib = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
off_conv_inhib = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
for t in range(TIMESTEPS):
    on_conv_inhib[t] = cv2.filter2D(on_inhib_lpf[t], -1, INHIB_KERNEL)
    off_conv_inhib[t] = cv2.filter2D(off_inhib_lpf[t], -1, INHIB_KERNEL)

# Visualise
fig, axes = plt.subplots(2,TIMESTEPS, sharey=True, figsize=(15, 5))
for t in range(TIMESTEPS):
    axes[0,t].imshow(on_conv_inhib[t,:,:], cmap="gray")
    axes[1,t].imshow(off_conv_inhib[t,:,:], cmap="gray")
    
    hide_axes(axes[0,t])
    hide_axes(axes[1,t])
fig.tight_layout(pad=0)
save_fig('inhibition.png')
imshow_single_vis([on_conv_inhib[fig_index[1],:,:], off_conv_inhib[fig_index[1],:,:]], title='inhibition_single.png')

#%%
"""
FDSR

"Fast depolarization, slow repolarization (FDSR). If the input signal is ‘depolarizing’ (positive temporal gradient), 
a first-order low pass filter with a small time constant (LPFfast) is used, otherwise for a ‘repolarizing’ signal 
(negative gradient) a larger time constant is applied (LPFslow). The resulting processed signal represents an 
‘adaptation state’ which then subtractively inhibits the unaltered pass-through signal"
"""

# Convert two time constants required for FDSR into decays to use in simple low-pass filters
FDSR_TAU_FAST = 1
FDSR_TAU_SLOW = 100
FSR_K_FAST = np.exp(-1 / FDSR_TAU_FAST)
FSR_K_SLOW = np.exp(-1 / FDSR_TAU_SLOW)

# Depending on instantaneous temporal gradients, pick K values for each pixel at each timestep
k_on = np.where((on_f[1:,:,:] - on_f[:-1,:,:]) >= 0.0, FSR_K_FAST, FSR_K_SLOW)
k_off = np.where((off_f[1:,:,:] - off_f[:-1,:,:]) > 0.0, FSR_K_FAST, FSR_K_SLOW)
    
# Apply low-pass filters to on and off channels
a_on = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
a_off = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
for t in range(1, TIMESTEPS):
    a_on[t] = ((1.0 - k_on[t - 1]) * on_f[t]) + (k_on[t - 1] * a_on[t - 1]) 
    a_off[t] = ((1.0 - k_off[t - 1]) * off_f[t]) + (k_off[t - 1] * a_off[t - 1]) 
    
visualise([on_f - a_on, off_f - a_off], ["On FDSR", "Off FDSR"], title='FDSR')
visualise([on_f - a_on - on_conv_inhib, off_f - a_off - off_conv_inhib], ["On after inhibition", "Off after inhibition"], title='After inhibition')

single_vis([on_f - a_on, off_f - a_off], ["On FDSR", "Off FDSR"], fig_index=fig_index, title='FDSR_single')
single_vis([on_f - a_on - on_conv_inhib, off_f - a_off - off_conv_inhib], ["On after inhibition", "Off after inhibition"], 
           fig_index=fig_index, title='After inhibition_single')

#%%
"""
Half-wave rectification
"""

# Subtract output of FDSR and inhibitory signal from each channel and rectify again 
# **NOTE** because we're only implementing one direction we only take positive channels
# def rectification_polarity(obj_colour):
#     if obj_colour == 'black': # black-on-white
#         a_HWR_on = np.maximum(0.0, on_f - a_on - on_conv_inhib) # ON of on
#         a_HWR_off = np.maximum(0.0, off_f - a_off - off_conv_inhib) # ON of off
#         # a_HWR_off = -np.minimum(0.0, off_f - a_off - off_conv_inhib) # OFF of off
#     else:
#         a_HWR_off = -np.minimum(0.0, on_f - a_on - on_conv_inhib) # OFF of on
#         a_HWR_on = np.maximum(0.0, off_f - a_off - off_conv_inhib) # ON of off
        
#     return a_HWR_on, a_HWR_off
        
# This results in OFF signal always being the one delayed later by LPF5
# a_on, a_off = rectification_polarity(obj_colour)

a_on = np.maximum(0.0, on_f - a_on - on_conv_inhib) # ON of on
a_off = np.maximum(0.0, off_f - a_off - off_conv_inhib) # ON of off

visualise([a_on, a_off], ["On_ON", "Off_OFF"], title='After HW-R before LP5')
single_vis([a_on, a_off], ["On_ON", "Off_OFF"], fig_index=fig_index, title='After HW-R before LP5_single')

#%%
"""
Recombine
"""

DELAY_TAU = 25
DELAY_K = np.exp(-1 / DELAY_TAU)
consts = {'a':1.0, 'b':1.0, 'c':1.0}

def mode(**kwargs):
    
    off_filter = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
    output = np.zeros((TIMESTEPS, HEIGHT, WIDTH, 3))
    
    if kwargs['mode'] == 'RTC':
        consts['c'] = 0
    elif kwargs['mode'] == 'ESTMD':
        consts['a'], consts['b'] = 0, 0
    else:
        pass
    
    # Correlate (multiply) low-pass filtered off channel with on channel        
    for t in range(1, TIMESTEPS):
        off_filter[t] = ((1.0 - DELAY_K) * a_off[t]) + (DELAY_K * off_filter[t - 1]) 
        output[t,:,:,0] = consts['b'] * off_filter[t] + consts['a'] * a_on[t] # RTC
        output[t,:,:,1] = consts['c'] * off_filter[t] * a_on[t] # ESTMD
        output[t,:,:,2] = output[t,:,:,0] + output[t,:,:,1] # OUTPUT
        
    return output, off_filter

output, off_filter = mode(mode='ESTMD', consts=consts)

visualise([a_on, off_filter], ["On_ON", "Off_OFF"], title='After LP5')
single_vis([a_on, off_filter], ["On_ON", "Off_OFF"], fig_index=fig_index, title='After LP5_single')

visualise([output[:,:,:,0], output[:,:,:,1], output[:,:,:,2]], ['\u03A3', 'X', '\u03A3 + X'], title='Circuit output')
single_vis([output[:,:,:,0], output[:,:,:,1], output[:,:,:,2]], ['\u03A3', 'X', '\u03A3 + X'], fig_index=fig_index, title='Circuit output_single')

if obj_colour == 'black':
    output = np.ones(output.shape) - output
else:
    pass

fig, axes = plt.subplots(1, TIMESTEPS, sharey=True, figsize=(15, 5))
for t, a in enumerate(axes):
    a.imshow(output[t,:,:,2], vmin=np.min(output[:,:,:,2]), vmax=np.max(output[:,:,:,2]), cmap="gray")
    hide_axes(a)
fig.tight_layout(pad=0)
save_fig('ESTMD output.png')
imshow_single_vis([output[fig_index[1],:,:,2]], vmin=np.min(output[:,:,:,2]), vmax=np.max(output[:,:,:,2]), title='ESTMD output_single.png')

t2 = time.perf_counter()
print(f'{t2-t1:.2f}')