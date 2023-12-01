import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import signal

# import TSDN_data_analysis

stimuli = ['alone', 'stationary', 'background_right', 'background_left']
background = 'sinusoidal'

def filtering(x):
    b, a = signal.iirfilter(1, Wn=1, fs=165, btype="low")
    return signal.filtfilt(b, a, x)

def stim_load(filename):
    return pd.read_csv(filename + f'_{background}.csv')

neurons = ['STMD_R', 'STMD_L', 'LPTC_R', 'LPTC_L', 'LPTC_HR']

pre_remaining = 5
multiplier = 1
post_remaining = 5

pre_stim = int(pre_remaining/165 * 1000)
stim_time = int(multiplier * 160/165 * 1000)
post_stim = int(post_remaining/165 * 1000)

stims = []
for stim in stimuli:
    df = stim_load(stim)
    filtered_neurons = []
    for n in neurons:
        # Quantise
        raw = df[n].to_numpy()
        quantised_stim = raw[pre_stim:pre_stim+stim_time][::multiplier]
        # Join back up
        quantised_total = np.concatenate((raw[:pre_stim], quantised_stim, raw[-post_stim:]))
        # Low-pass filter
        filtered = filtering(quantised_total)
        filtered_neurons.append(filtered)
    stims.append(np.vstack(filtered_neurons))
    
stims = np.stack(stims) # (stimuli, neurons, timesteps)

sdfs = pickle.load(open(f'sdfs_{background}.pkl', 'rb'))

def model_D(param, neurons):
    a, c, d = param
    STMD_R, LPTC_R, LPTC_L = neurons
    
    return (a*STMD_R - (c*LPTC_R - d*LPTC_L).clip(min=0)).clip(min=0)

def model_E(param, neurons):
    a, e = param
    STMD_R, LPTC_HR = neurons
    
    return (a*STMD_R - e*LPTC_HR).clip(min=0)

def model_F(param, neurons):
    a, c, d = param
    STMD_R, LPTC_R, LPTC_L = neurons
    
    return (a*STMD_R*(1 + d*LPTC_L) - c*LPTC_R).clip(min=0)

def model_G(param, neurons):
    a, b, c = param
    STMD_R, STMD_L, LPTC_R = neurons
    
    return (a*STMD_R*(1 + b*STMD_L) - c*LPTC_R).clip(min=0)

def model_H(param, neurons):
    a, c, d = param
    STMD_R, LPTC_R, LPTC_L = neurons
    
    return (a*STMD_R - c*LPTC_R + d*LPTC_L).clip(min=0)

def simulate_shift(var, shift):
    return np.roll(var.copy(), shift)

fig, axes = plt.subplots(5, figsize=(6, 12), dpi=500, sharex=True, sharey=True)
for j, ax in enumerate(axes):
    err = 0
    for i in range(len(stimuli)):
        STMD_R = stims[i][neurons.index('STMD_R')].clip(min=0)
        STMD_L = stims[i][neurons.index('STMD_L')].clip(min=0)
        LPTC_R = stims[i][neurons.index('LPTC_R')].clip(min=0)
        LPTC_L = stims[i][neurons.index('LPTC_L')].clip(min=0)
        LPTC_HR = stims[i][neurons.index('LPTC_HR')]
        
        # fig, axes = plt.subplots(dpi=500)
        # axes.plot(STMD_R.clip(min=0), label='STMD_R')
        # axes.plot(STMD_L.clip(min=0), label='STMD_L')
        # axes.plot(LPTC_R.clip(min=0), label='LPTC_R')
        # axes.plot(LPTC_L.clip(min=0), label='LPTC_L')
        # axes.plot(LPTC_HR, label='LPTC_HR')
        # plt.legend()
        
        TSDN_ephys = np.mean(np.vstack(sdfs[i]), axis=0)[1:-1]
        
        shift = np.argmax(TSDN_ephys) - np.argmax(STMD_R)
        TSDN_ephys = simulate_shift(TSDN_ephys, -shift)
        
        if j == 0:
            title = 'D'
            sim = model_D([51, 1.03, 0.39], [STMD_R, LPTC_R, LPTC_L])
        elif j == 1:
            title = 'E'
            sim = model_E([22.5, 0.015], [STMD_R, LPTC_HR])
        elif j == 2:
            title = 'F'
            sim = model_F([9, 0.01, 1.225], [STMD_R, LPTC_R, LPTC_L])
        elif j == 3:
            title = 'G'
            sim = model_G([24, 1.4, 0.002], [STMD_R, STMD_L, LPTC_R])
        elif j == 4:
            title = 'H'
            sim = model_H([24.52, 0.026, 0.017], [STMD_R, LPTC_R, LPTC_L])
            
        err += (np.sum((sim - TSDN_ephys)**2) / len(TSDN_ephys))
            
        stimuli[i] = stimuli[i].replace('_', ' ')
            
        axes[j].text(0.05, 0.8, title, horizontalalignment='left', transform=ax.transAxes)
        axes[j].plot(sim, label=stimuli[i])
    print(title, np.sqrt(err / stims.shape[-1])) # RMSE
axes[-1].set_xlabel('Time [ms]')
axes[-1].set_ylabel('Activation [a.u.]')
plt.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Each plot showing all model results for each condition
fig, axes = plt.subplots(len(stimuli), figsize=(6, 10), dpi=500, sharex=True, sharey=True)
for i, ax in enumerate(axes):
    STMD_R = stims[i][neurons.index('STMD_R')].clip(min=0)
    STMD_L = stims[i][neurons.index('STMD_L')].clip(min=0)
    LPTC_R = stims[i][neurons.index('LPTC_R')].clip(min=0)
    LPTC_L = stims[i][neurons.index('LPTC_L')].clip(min=0)
    LPTC_HR = stims[i][neurons.index('LPTC_HR')]
    
    # fig, axes = plt.subplots(dpi=500)
    # axes.plot(STMD_R.clip(min=0), label='STMD_R')
    # axes.plot(STMD_L.clip(min=0), label='STMD_L')
    # axes.plot(LPTC_R.clip(min=0), label='LPTC_R')
    # axes.plot(LPTC_L.clip(min=0), label='LPTC_L')
    # axes.plot(LPTC_HR, label='LPTC_HR')
    # plt.legend()
    
    TSDN_ephys = np.mean(np.vstack(sdfs[i]), axis=0)[1:-1]
    
    shift = np.argmax(TSDN_ephys) - np.argmax(STMD_R)
    TSDN_ephys = simulate_shift(TSDN_ephys, -shift)
    
    ax.plot(model_D([51, 1.03, 0.39], [STMD_R, LPTC_R, LPTC_L]), label='D')
    ax.plot(model_E([22.5, 0.015], [STMD_R, LPTC_HR]), label='E')
    ax.plot(model_F([9, 0.01, 1.225], [STMD_R, LPTC_R, LPTC_L]), label='F')
    ax.plot(model_G([24, 1.4, 0.002], [STMD_R, STMD_L, LPTC_R]), label='G')
    ax.plot(model_H([24.52, 0.026, 0.017], [STMD_R, LPTC_R, LPTC_L]), label='H')
        
    stimuli[i] = stimuli[i].replace('_', ' ')
        
    ax.text(0.05, 0.8, stimuli[i], horizontalalignment='left', transform=ax.transAxes)
    ax.plot(TSDN_ephys, label='TSDN ephys')
axes[-1].set_xlabel('Time [ms]')
axes[-1].set_ylabel('Activation [a.u.]')
plt.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
            
        
    # For quick check -- model H
    # i = 0
    # STMD_R = stims[i][neurons.index('STMD_R')].clip(min=0)
    # STMD_L = stims[i][neurons.index('STMD_L')].clip(min=0)
    # LPTC_R = stims[i][neurons.index('LPTC_R')].clip(min=0)
    # LPTC_L = stims[i][neurons.index('LPTC_L')].clip(min=0)
    # LPTC_HR = stims[i][neurons.index('LPTC_HR')]

    # TSDN_ephys = np.mean(np.vstack(sdfs[i]), axis=0)[1:-1]
    # shift = np.argmax(TSDN_ephys) - np.argmax(STMD_R)
    # TSDN_ephys = simulate_shift(TSDN_ephys, -shift)

    # fig, axes = plt.subplots(dpi=500)
    # a=30; c=0.05; d=0.05
    # axes.plot(a*STMD_R, label='STMD')
    # axes.plot(LPTC_R, label='LPTC_R')
    # axes.plot(d*LPTC_L, label='LPTC_L')
    # axes.plot((a*STMD_R - c*LPTC_R + d*LPTC_L).clip(min=0), label='TSDN')
    # axes.plot(TSDN_ephys, label='TSDN ephys')
    # plt.legend()
    
    # if args.model_type == 'D':
    #     TSDN = model_D(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
    # elif args.model_type == 'E':
    #     TSDN = model_E(params, [STMD_R, LPTC_HR])[:len(TSDN_ephys)]
    # elif args.model_type == 'F':
    #     TSDN = model_F(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
    # elif args.model_type == 'G':
    #     TSDN = model_G(params, [STMD_R, STMD_L, LPTC_R])[:len(TSDN_ephys)]
    # elif args.model_type == 'H':
    #     TSDN = model_H(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
    # else:
    #     print('Unknown model type')