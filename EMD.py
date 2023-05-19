"""
Adapted from Bagheri, Z.M., Wiederman, S.D., Cazzolato, B.S., Grainger, S. and O'Carroll, D.C., 2017. 
Performance of an insect-inspired target tracker in natural conditions. Bioinspiration & biomimetics, 12(2), p.025006.

Eichner, H., Joesch, M., Schnell, B., Reiff, D.F. and Borst, A., 2011. 
Internal structure of the fly elementary motion detector. Neuron, 70(6), pp.1155-1164.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

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

def save_fig(title):
    save_out = os.path.join(os.getcwd(), 'EMD_figs')
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title))
    plt.close()

class EMD:
    def __init__(self, **kwargs):
        self.HEIGHT = 16
        self.WIDTH = 16
        self.TIMESTEPS = 16
        self.VELOCITY = 1

        self.STIM_HEIGHT = 2
        self.STIM_WIDTH = 2
        self.STIM_Y = 2
        
        self.direction = kwargs['DIRECTION']
        
        # Time constants
        self.HPF_TAU = 40
        self.LPF_TAU = 35
        
        # Decay terms
        self.HPF_K = np.exp(-1 / self.HPF_TAU)
        self.LPF = np.exp(-1 / self.LPF_TAU)
        
    def image_generation(self):
        # Initialise images tensor
        images = np.ones((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        for i in range(self.TIMESTEPS):
            start_pos_x = i*self.VELOCITY
            images[i,self.STIM_Y:self.STIM_Y+self.STIM_HEIGHT,start_pos_x:start_pos_x+self.STIM_WIDTH] = 0.0
        if self.direction == 'RIGHT':
            pass
        else:
            images = images[::-1,:,:]
        return images
        
    def model(self):
        images = self.image_generation()
        # Initialisations

        hpf = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        on_f = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        off_f = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        a_on = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        a_off = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        output = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH-1))

        for t in range(1, self.TIMESTEPS):
            # High-pass filters
            hpf[t] = ((1.0 - self.HPF_K) * hpf[t - 1]) + ((1.0 - self.HPF_K) * (images[t] - images[t - 1]))
            
            # Half-wave rectified channels
            on_f[t] = np.maximum(hpf[t], 0.0)
            off_f[t] = -np.minimum(hpf[t], 0.0)
            
            # Low-pass filtered delayed channels
            a_on[t] = ((1.0 - self.LPF) * on_f[t]) + (self.LPF * a_on[t - 1]) 
            a_off[t] = ((1.0 - self.LPF) * off_f[t]) + (self.LPF * a_off[t - 1]) 
            
            # Correlate
            on_out = (a_on[t,:,:-1] * on_f[t,:,1:]) - (on_f[t,:,:-1] * a_on[t,:,1:])
            off_out = (a_off[t,:,:-1] * off_f[t,:,1:]) - (off_f[t,:,:-1] * a_off[t,:,1:])
            
            # Summation
            output[t] = on_out + off_out
        
        self.outs = {'hpf':hpf, 'on_f':on_f, 'off_f':off_f, 'a_on':a_on, 'a_off':a_off, 'output':output}    
        
        return self.outs

    def visualisations(self):
        images = self.image_generation()
        
        # Visualize stimuli
        fig, axes = plt.subplots(1,self.TIMESTEPS, sharey=True, figsize=(15,5))
        for t, a in enumerate(axes):
            a.imshow(images[t,:,:], cmap="gray", vmin=0.0, vmax=1.0)
            hide_axes(a)
        fig.tight_layout(pad=0)
        save_fig('stimuli.png')
        
        visualise([self.outs['on_f'], self.outs['off_f']], ['On', 'Off'], title='Rectified channels')
        visualise([self.outs['a_on'], self.outs['a_off']], ['On', 'Off'], title='Low-pass filtered channels')
        visualise([self.outs['output']], title=f'Output_{self.direction}')
        
EMD_model = EMD(DIRECTION='RIGHT')
outs = EMD_model.model()
EMD_model.visualisations()