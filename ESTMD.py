"""
Adapted from Jamie's code available at https://github.com/neworderofjamie/hover
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import mode

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
    save_out = os.path.join(os.getcwd(), 'figs', save_folder)
    os.makedirs(save_out, exist_ok=True)
    
    plt.savefig(os.path.join(save_out, title))
    plt.close()

# Single panel visualisation
def single_vis(vars, labels=None, **kwargs):
    fig, axes = plt.subplots(figsize=(15,10))
    _, row, col = kwargs['fig_index']
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

class ESTMD:
    def __init__(self, obj_colour, MULTI_TARGET=None, **kwargs):
        # Resolution of ommatidia array and number of timesteps to simulate
        self.WIDTH = kwargs['WIDTH']
        self.HEIGHT = kwargs['HEIGHT']
        self.TIMESTEPS = kwargs['TIMESTEPS']
        
        # Size of stimuli, its y coordinate and its velocity 
        self.STIM_WIDTH = kwargs['STIM_WIDTH']
        self.STIM_HEIGHT = kwargs['STIM_HEIGHT']
        self.STIM_Y = kwargs['STIM_Y']
        self.VELOCITY = kwargs['VELOCITY']
        
        self.mode = kwargs['MODE']
        if self.mode == 'RTC':
            self.RTC_TEST = kwargs['RTC_TEST']
        else:
            self.RTC_TEST = None
        if MULTI_TARGET is not None:
            self.FLAG_M_TARGETS = 1
        self.obj_colour = obj_colour
        self.OMMATIDIA_COL = kwargs['OMMATIDIA_COL']
        
        # Time constants
        self.HPF_TAU = 40.0
        self.INHIB_LPF_TAU = 2.0
        # Define 1D kernel
        self.INHIB_KERNEL = (2/25) * np.asarray([[-1, -1, -1, -1, -1],
                                                 [-1, -1, -1, -1, -1],
                                                 [-1, -1, 2, -1, -1],
                                                 [-1, -1, -1, -1, -1],
                                                 [-1, -1, -1, -1, -1]], dtype=float) # 0.2 * np.asarray([2.0, 1.0, 0.0, 1.0, 2.0])
        self.FDSR_TAU_FAST = 3.0
        self.FDSR_TAU_SLOW = 70.0
        self.DELAY_TAU = 25.0
        
        # Decay terms
        self.HPF_K = np.exp(-1 / self.HPF_TAU)
        self.INHIB_LPF_K = np.exp(-1 / self.INHIB_LPF_TAU)
        self.FDSR_K_FAST = np.exp(-1 / self.FDSR_TAU_FAST)
        self.FDSR_K_SLOW = np.exp(-1 / self.FDSR_TAU_SLOW)
        self.DELAY_K = np.exp(-1 / self.DELAY_TAU)
        
    def image_generation(self):
        # Generate TIMESTEPS long 'video' of stimuli moving across frame
        if self.mode == 'RTC' and self.RTC_TEST is not None:
            self.save_folder = '/'.join(['RTC', self.RTC_TEST])
            
            self.PULSE_WIDTH = 5 # Pulse width
            
            if self.RTC_TEST == 'HIGH':
                INTERVAL = 10
                self.TIMESTEPS = 90
            else:
                # Interval between pulses
                INTERVAL = 30
            # self.TIMESTEPS = (INTERVAL + self.PULSE_WIDTH) * 20 + 1 # Timestep to see adaptation
                
            images = np.ones((self.TIMESTEPS, self.HEIGHT, self.WIDTH)) * 0.5

            for i in range(self.TIMESTEPS-1):
                if i <= self.TIMESTEPS / 2:
                    if not i % (self.PULSE_WIDTH + INTERVAL):
                        for k in range(self.PULSE_WIDTH):
                            images[i+k+1,:,:] = 1.0
                else:
                    if not i % (self.PULSE_WIDTH + INTERVAL):
                        for k in range(self.PULSE_WIDTH):
                            images[i+k+1,:,:] = 0.0
                            # images[i+k,self.STIM_Y:self.STIM_Y+self.STIM_HEIGHT,self.STIM_Y:self.STIM_Y+self.STIM_WIDTH] = 0.0
        else:
            self.save_folder = 'ESTMD'
            images = np.ones((self.TIMESTEPS, self.HEIGHT, self.WIDTH))

            for i in range(self.TIMESTEPS):
                start_pos_x = i*self.VELOCITY
                images[i,self.STIM_Y:self.STIM_Y+self.STIM_HEIGHT,start_pos_x:start_pos_x+self.STIM_WIDTH] = 0.0
                
            if self.obj_colour == 'black': # black-on-white
                pass
            else:
                images = np.ones(images.shape) - images
        global save_folder
        save_folder = self.save_folder
            
        return images
    
    def colour_assertion(self, images, TIMESTEP):
        # Check if stimulus is brighter or darker than background
        self.fig_index = (self.STIM_Y, self.OMMATIDIA_COL)
        self.fig_index = (TIMESTEP, *self.fig_index)
        if self.mode == 'RTC':
            if np.unique(mode(images[TIMESTEP,:,:], keepdims=True))[0] > 0.5:
                self.obj_colour = 'white'
            elif np.unique(mode(images[TIMESTEP,:,:], keepdims=True))[0] < 0.5:
                self.obj_colour = 'black'
            else:
                pass
        else:
            pass
    
    def model_mode(self, consts):
        if self.mode == 'RTC':
            consts['c'] = 0
        elif self.mode == 'ESTMD':
            consts['a'], consts['b'] = 0, 0
        else:
            pass
        return consts
        
    def model(self):
        images = self.image_generation()
        
        # Initialisations
        hpf = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        on_f = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        off_f = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        on_inhib_lpf = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        off_inhib_lpf = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        on_conv_inhib = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        off_conv_inhib = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        a_on = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        a_off = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        a_on_rect = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        a_off_rect = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        
        self.on_filter = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        self.off_filter = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH))
        output = np.zeros((self.TIMESTEPS, self.HEIGHT, self.WIDTH, 3))
        
        # Recombine
        consts = {'a':1.0, 'b':1.0, 'c':1.0}
        consts = self.model_mode(consts)
        
        for t in range(1, self.TIMESTEPS):
            # High-pass filters, HP3
            hpf[t] = ((1.0 - self.HPF_K) * hpf[t - 1]) + ((1.0 - self.HPF_K) * (images[t] - images[t - 1]))
            
            # Half-wave rectification
            # Clamp the high pass filtered data to separate the on and off channels
            on_f[t] = np.maximum(hpf[t], 0.0)
            off_f[t] = -np.minimum(hpf[t], 0.0)
            
            # Low-pass filtering, LP4
            # Convert time constant of low pass filter into a decay to use in a simple low-pass filter
            # Apply low-pass filter to on and off channel
            on_inhib_lpf[t] = (self.INHIB_LPF_K * on_inhib_lpf[t - 1]) + ((1.0 - self.INHIB_LPF_K) * on_f[t])
            off_inhib_lpf[t] = (self.INHIB_LPF_K * off_inhib_lpf[t - 1]) + ((1.0 - self.INHIB_LPF_K) * off_f[t])
            
            # Inhibition
            # Convolve the 1D kernel with low-pass filtered input every timestep
            on_conv_inhib[t] = cv2.filter2D(on_inhib_lpf[t], -1, self.INHIB_KERNEL)
            off_conv_inhib[t] = cv2.filter2D(off_inhib_lpf[t], -1, self.INHIB_KERNEL)
            # on_conv_inhib[t] = cv2.filter2D(on_f[t], -1, self.INHIB_KERNEL)
            # off_conv_inhib[t] = cv2.filter2D(off_f[t], -1, self.INHIB_KERNEL)
            
            # "Fast depolarization, slow repolarization (FDSR). If the input signal is ‘depolarizing’ 
            # (positive temporal gradient), a first-order low pass filter with a small time constant 
            # (LPFfast) is used, otherwise for a ‘repolarizing’ signal (negative gradient) a larger 
            # time constant is applied (LPFslow). The resulting processed signal represents an 
            # ‘adaptation state’ which then subtractively inhibits the unaltered pass-through signal"
            
            # Depending on instantaneous temporal gradients, pick K values for each pixel at each timestep
            k_on = np.where((on_f[t,:,:] - on_f[t-1,:,:]) > 0.0, self.FDSR_K_FAST, self.FDSR_K_SLOW)
            k_off = np.where((off_f[t,:,:] - off_f[t-1,:,:]) > 0.0, self.FDSR_K_FAST, self.FDSR_K_SLOW)
            # Doesn't match Weiderman paper but is the only thing that makes sense
            
            # Apply low-pass filters to on and off channels
            a_on[t] = ((1.0 - k_on) * on_f[t]) + (k_on * a_on[t - 1]) 
            a_off[t] = ((1.0 - k_off) * off_f[t]) + (k_off * a_off[t - 1]) 
            
            # Half-wave rectification
            a_on_rect[t] = np.maximum(0.0, on_f[t] - a_on[t] + on_conv_inhib[t]) # ON of on
            a_off_rect[t] = np.maximum(0.0, off_f[t] - a_off[t] + off_conv_inhib[t]) # ON of off
            
            # Correlate (multiply) low-pass filtered off channel with on channel
            # if self.obj_colour == 'black':
            #     self.off_filter[t] = ((1.0 - self.DELAY_K) * a_off_rect[t]) + (self.DELAY_K * self.off_filter[t - 1]) 
            #     output[t,:,:,0] = consts['b'] * self.off_filter[t] + consts['a'] * a_on_rect[t] # RTC
            #     output[t,:,:,1] = consts['c'] * self.off_filter[t] * a_on_rect[t] # ESTMD
            # elif self.obj_colour == 'white':
            #     self.on_filter[t] = ((1.0 - self.DELAY_K) * a_on_rect[t]) + (self.DELAY_K * self.on_filter[t - 1]) 
            #     output[t,:,:,0] = consts['b'] * self.on_filter[t] + consts['a'] * a_off_rect[t] # RTC
            #     output[t,:,:,1] = consts['c'] * self.on_filter[t] * a_off_rect[t] # ESTMD
            
            # Sensitivities based on target vs. background contrast
            self.colour_assertion(images, t)
            if self.obj_colour == 'black':
                self.off_filter[t] = ((1.0 - self.DELAY_K) * a_off_rect[t]) + (self.DELAY_K * self.off_filter[t - 1]) 
                output[t,:,:,0] = consts['c'] * self.off_filter[t] * a_on_rect[t] # ESTMD
                output[t,:,:,1] = (consts['b'] * self.off_filter[t]) + (consts['a'] * a_on_rect[t]) # RTC
            else:
                self.on_filter[t] = ((1.0 - self.DELAY_K) * a_on_rect[t]) + (self.DELAY_K * self.on_filter[t - 1]) 
                output[t,:,:,0] = consts['c'] * self.on_filter[t] * a_off_rect[t] # ESTMD
                output[t,:,:,1] = (consts['b'] * self.on_filter[t]) + (consts['a'] * a_off_rect[t]) # RTC
            
            output[t,:,:,2] = output[t,:,:,0] + output[t,:,:,1] # OUTPUT
        
        self.outs = {'hpf':hpf, 'on_f':on_f, 'off_f':off_f, 'on_inhib_lpf':on_inhib_lpf, 'off_inhib_lpf':off_inhib_lpf,
                'on_conv_inhib':on_conv_inhib, 'off_conv_inhib':off_conv_inhib, 'a_on':a_on, 'a_off':a_off, 
                'a_on_rect':a_on_rect, 'a_off_rect':a_off_rect, 'output':output}
        
        return self.outs

    def visualisations(self):
        # # visualise([self.outs['hpf']], title='HP3')
        # # single_vis([self.outs['hpf']], fig_index=self.fig_index, title='HP3_single')
        
        # Visualize stimuli
        images = self.image_generation()
        fig, axes = plt.subplots(1,self.TIMESTEPS, sharey=True, figsize=(15,5))
        for t, a in enumerate(axes):
            a.imshow(images[t,:,:], cmap="gray", vmin=0.0, vmax=1.0)
            hide_axes(a)
        fig.tight_layout(pad=0)
        save_fig('stimuli.png')
        imshow_single_vis([images[self.fig_index[1],:,:]], title='stimuli_single.png')
        
        # # After HPF3
        # visualise([self.outs['on_f'], self.outs['off_f']], ["On", "Off"], title='HW-R after HPF3')
        # single_vis([self.outs['on_f'], self.outs['off_f']], ["On", "Off"], fig_index=self.fig_index, title='HW-R after HPF3_single')
        
        # # LP4
        # visualise([self.outs['on_inhib_lpf'], self.outs['off_inhib_lpf']], ["On LPF", "Off LPF"], title='LP4')
        # single_vis([self.outs['on_inhib_lpf'], self.outs['off_inhib_lpf']], ["On LPF", "Off LPF"], fig_index=self.fig_index, title='LP4_single')
        
        # # Inhibition
        # fig, axes = plt.subplots(2,self.TIMESTEPS, sharey=True, figsize=(15, 5))
        # for t in range(self.TIMESTEPS):
        #     axes[0,t].imshow(self.outs['on_conv_inhib'][t,:,:], cmap="gray", 
        #                       vmin=np.min(self.outs['on_conv_inhib']), vmax=np.max(self.outs['on_conv_inhib']))
        #     axes[1,t].imshow(self.outs['off_conv_inhib'][t,:,:], cmap="gray", 
        #                       vmin=np.min(self.outs['off_conv_inhib']), vmax=np.max(self.outs['off_conv_inhib']))
            
        #     hide_axes(axes[0,t])
        #     hide_axes(axes[1,t])
        # fig.tight_layout(pad=0)
        # save_fig('inhibition.png')
        # imshow_single_vis([self.outs['on_conv_inhib'][self.fig_index[1],:,:], 
        #                     self.outs['off_conv_inhib'][self.fig_index[1],:,:]], 
        #                   vmin=np.min(np.minimum(self.outs['on_conv_inhib'][:,:,:], 
        #                                   self.outs['off_conv_inhib'][:,:,:])), 
        #                   vmax=np.max(np.maximum(self.outs['on_conv_inhib'][:,:,:], 
        #                                   self.outs['off_conv_inhib'][:,:,:])), 
        #                   title='inhibition_single.png')
        
        # single_vis([self.outs['on_conv_inhib'], self.outs['off_conv_inhib']], 
        #             ["On Inhibition", "Off Inhibition"], 
        #             fig_index=self.fig_index, title='Inhibition_plot_single')
        
        # # After FDSR
        # visualise([self.outs['on_f'] - self.outs['a_on'], self.outs['off_f'] - self.outs['a_off']], ["On FDSR", "Off FDSR"], title='FDSR')
        
        # # After inhibition
        # visualise([self.outs['on_f'] - self.outs['a_on'] + self.outs['on_conv_inhib'], 
        #             self.outs['off_f'] - self.outs['a_off'] + self.outs['off_conv_inhib']], ["On after inhibition", "Off after inhibition"], 
        #           title='After inhibition')
        
        # single_vis([self.outs['on_f'] - self.outs['a_on'] + self.outs['on_conv_inhib'], 
        #             self.outs['off_f'] - self.outs['a_off'] + self.outs['off_conv_inhib']], 
        #             ["On after inhibition", "Off after inhibition"], 
        #             fig_index=self.fig_index, title='After inhibition_single')
        
        # single_vis([self.outs['a_on'], self.outs['a_off']], 
        #             ["On FDSR", "Off FDSR"], 
        #             fig_index=self.fig_index, title='FDSR_single')
        
        # single_vis([self.outs['on_f'] - self.outs['a_on'], self.outs['off_f'] - self.outs['a_off']], 
        #             ["On FDSR", "Off FDSR"], 
        #             fig_index=self.fig_index, title='After FDSR_single')
        
        # # After HW-R2 but before LPF5
        # visualise([self.outs['a_on_rect'], self.outs['a_off_rect']], ["On_ON", "Off_OFF"], title='After HW-R before LP5')
        # single_vis([self.outs['a_on_rect'], self.outs['a_off_rect']], ["On_ON", "Off_OFF"], fig_index=self.fig_index, 
        #             title='After HW-R before LP5_single')
        
        # # After LPF5
        # if not self.mode == 'RTC':
        #     if self.obj_colour == 'black':
        #         visualise([self.outs['a_on_rect'], self.off_filter], ["On_ON", "Off_OFF"], title='After LP5')
        #         single_vis([self.outs['a_on_rect'], self.off_filter], ["On_ON", "Off_OFF"], fig_index=self.fig_index, 
        #                     title=f'After LP5_single_{self.obj_colour}')
        #     elif self.obj_colour == 'white':
        #         visualise([self.on_filter, self.outs['a_off_rect']], ["On_ON", "Off_OFF"], title='After LP5')
        #         single_vis([self.on_filter, self.outs['a_off_rect']], ["On_ON", "Off_OFF"], fig_index=self.fig_index, 
        #                     title=f'After LP5_single_{self.obj_colour}')
        # else:
        #     visualise([self.outs['a_on_rect'], self.off_filter], ["On_ON", "Off_OFF"], title='After LP5_off_delayed')
        #     single_vis([self.outs['a_on_rect'], self.off_filter], ["On_ON", "Off_OFF"], fig_index=self.fig_index, 
        #                 title='After LP5_single_off_delayed')
            
        #     visualise([self.on_filter, self.outs['a_off_rect']], ["On_ON", "Off_OFF"], title='After LP5_on_delayed')
        #     single_vis([self.on_filter, self.outs['a_off_rect']], ["On_ON", "Off_OFF"], fig_index=self.fig_index, 
        #                 title='After LP5_single_on_delayed')
        
        # # Circuit output
        # visualise([self.outs['output'][:,:,:,0], self.outs['output'][:,:,:,1], self.outs['output'][:,:,:,2]], 
        #           ['\u03A3', 'X', '\u03A3 + X'], title='Circuit output')
        # single_vis([self.outs['output'][:,:,:,0], self.outs['output'][:,:,:,1], self.outs['output'][:,:,:,2]], 
        #             ['\u03A3', 'X', '\u03A3 + X'], fig_index=self.fig_index, title='Circuit output_single')
        
        # ESTMD output
        if self.obj_colour == 'black':
            if np.max(self.outs['output']) > 0.0:
                self.outs['output'] = np.ones(self.outs['output'].shape) - self.outs['output']
        else:
            pass
        vmax_ = np.max(self.outs['output'][:,:,:,2])
        vmin_ = np.min(self.outs['output'][:,:,:,2])
        if (self.STIM_WIDTH or self.STIM_WIDTH) > 4:
            vmax_ = 1.0
            vmin_ = 0.0
        
        fig, axes = plt.subplots(1, self.TIMESTEPS, sharey=True, figsize=(15, 5))
        for t, a in enumerate(axes):
            a.imshow(self.outs['output'][t,:,:,2], 
                      vmin=vmin_, 
                      vmax=vmax_, 
                      cmap="gray")
            hide_axes(a)
        fig.tight_layout(pad=0)
        save_fig('ESTMD output.png')
        imshow_single_vis([self.outs['output'][self.fig_index[1],:,:,2]], 
                          vmin=vmin_, 
                          vmax=vmax_, 
                          title='ESTMD output_single.png')
        
    def figures_normalised(self, vars, labels=None):
        for i, v in enumerate(vars):
            v /= np.max(v)
            fig, axes = plt.subplots()
            plt.plot(v)
            if labels is not None:
                axes.set_xlabel(labels[i])
            save_fig(labels[i].split(' ',1)[0])

def height_test():
    outputs = []
    for i in range(23):
        ESTMD_model = ESTMD('black', 
                            OMMATIDIA_COL=4, 
                            WIDTH=25, 
                            HEIGHT=25, 
                            TIMESTEPS=25, 
                            STIM_WIDTH=4, 
                            STIM_HEIGHT=i, 
                            STIM_Y=1,
                            VELOCITY=1,
                            MODE='ESTMD')
    
        outs = ESTMD_model.model()
        outputs.append(np.max(outs['output'][:,:,:,2]))
    ESTMD_model.figures_normalised([outputs], ['Height (px)'])
    
def width_test():
    outputs = []
    for i in range(23):
        ESTMD_model = ESTMD('black', 
                            OMMATIDIA_COL=7, 
                            WIDTH=25, 
                            HEIGHT=25, 
                            TIMESTEPS=25, 
                            STIM_WIDTH=i, 
                            STIM_HEIGHT=4, 
                            STIM_Y=4,
                            VELOCITY=1,
                            MODE='ESTMD')
    
        outs = ESTMD_model.model()
        outputs.append(np.max(outs['output'][:,:,:,2]))
    ESTMD_model.figures_normalised([outputs], ['Width (px)'])
    
def velocity_test():
    outputs = []
    for i in range(23):
        ESTMD_model = ESTMD('black', 
                            OMMATIDIA_COL=7, 
                            WIDTH=25, 
                            HEIGHT=25, 
                            TIMESTEPS=25, 
                            STIM_WIDTH=4, 
                            STIM_HEIGHT=4, 
                            STIM_Y=4,
                            VELOCITY=i,
                            MODE='ESTMD')
    
        outs = ESTMD_model.model()
        outputs.append(np.max(outs['output'][:,:,:,2]))
    ESTMD_model.figures_normalised([outputs], ['Velocity (px/ms)'])

def RTC_test():
    for i in ['HIGH', 'LOW']:
        ESTMD_model = ESTMD('black', 
                            OMMATIDIA_COL=7, 
                            WIDTH=25, 
                            HEIGHT=25, 
                            TIMESTEPS=25, # Each timestep is 1 ms
                            STIM_WIDTH=4, 
                            STIM_HEIGHT=4, 
                            STIM_Y=4,
                            VELOCITY=1,
                            MODE='RTC',
                            RTC_TEST=i)
    
        outs = ESTMD_model.model()
        outs_ = outs['output'][:,:,:,2]
        r = []
        for t in range(outs_.shape[0]):
            r.append(np.max(outs_[t,:,:]))
        ESTMD_model.figures_normalised([r], [f'{i}_pulse_{ESTMD_model.PULSE_WIDTH}_ms_width'])
        if i == 'HIGH':
            ESTMD_model.visualisations()
            
def normal_test():
    ESTMD_model = ESTMD('black', 
                        OMMATIDIA_COL=7, 
                        WIDTH=25, 
                        HEIGHT=25, 
                        TIMESTEPS=25, # Each timestep is 1 ms
                        STIM_WIDTH=4, 
                        STIM_HEIGHT=4, 
                        STIM_Y=4,
                        VELOCITY=1,
                        MODE='RTC',
                        RTC_TEST='HIGH')

    ESTMD_model.model()
    ESTMD_model.visualisations()

def tests():    
    RTC_test()
    # normal_test()
    # # height_test()
    # width_test()
    # velocity_test()
    
tests()