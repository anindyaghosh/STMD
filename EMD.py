"""
Adapted from Bagheri, Z.M., Wiederman, S.D., Cazzolato, B.S., Grainger, S. and O’Carroll, D.C., 2017. 
Performance of an insect-inspired target tracker in natural conditions. Bioinspiration & biomimetics, 12(2), p.025006.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

HEIGHT=16
WIDTH=16
TIMESTEPS=16
VELOCITY=1

STIM_HEIGHT=2
STIM_WIDTH=2
STIM_Y=2

images = np.ones((TIMESTEPS, HEIGHT, WIDTH))

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

def save_fig(title):
    save_out = os.path.join(os.getcwd(), 'EMD_figs')
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title))
    plt.close()

def image_generation(direction):
    for i in range(TIMESTEPS):
        start_pos_x = i*VELOCITY
        if direction == 'right':
            images[i,STIM_Y:STIM_Y+STIM_HEIGHT,start_pos_x:start_pos_x+STIM_WIDTH] = 0.0
        else:
            images[i,STIM_Y:STIM_Y+STIM_HEIGHT,WIDTH-STIM_WIDTH-start_pos_x:WIDTH-start_pos_x] = 0.0
    return images

#%%    
hpf = np.zeros((TIMESTEPS, HEIGHT, WIDTH))

on_f = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
off_f = np.zeros((TIMESTEPS, HEIGHT, WIDTH))

a_on = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
a_off = np.zeros((TIMESTEPS, HEIGHT, WIDTH))

on_out = np.zeros((TIMESTEPS, HEIGHT, WIDTH))
off_out = np.zeros((TIMESTEPS, HEIGHT, WIDTH))

output = np.zeros((TIMESTEPS, HEIGHT, WIDTH))

HPF_TAU = 40
LPF_TAU = 35

HPF_K = np.exp(-1 / HPF_TAU)
LPF = np.exp(-1 / LPF_TAU)

images = image_generation('left')

for t in range(1, TIMESTEPS):
    # High-pass filters
    hpf[t] = ((1.0 - HPF_K) * hpf[t - 1]) + ((1.0 - HPF_K) * (images[t] - images[t - 1]))
    
    # Half-wave rectified channels
    on_f[t] = np.maximum(hpf[t], 0.0)
    off_f[t] = -np.minimum(hpf[t], 0.0)
    
    # Low-pass filtered delayed channels
    a_on[t] = ((1.0 - LPF) * on_f[t]) + (LPF * a_on[t - 1]) 
    a_off[t] = ((1.0 - LPF) * off_f[t]) + (LPF * a_off[t - 1]) 
    
    # Correlate
    on_out[t] = off_f[t] * a_on[t]
    off_out[t] = on_f[t] * a_off[t]
    
    # Summation
    output[t] = off_out[t] - on_out[t]
    
visualise([on_f, off_f], ['On', 'Off'], title='Rectified channels')
visualise([a_on, a_off], ['On', 'Off'], title='Low-pass filtered channels')
visualise([on_out, off_out], ['On', 'Off'], title='Correlation')
visualise([output], title='Output')