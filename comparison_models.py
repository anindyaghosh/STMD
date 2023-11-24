import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import signal

# import TSDN_data_analysis

stimuli = ['alone', 'stationary', 'background_right', 'background_left']

def filtering(x):
    b, a = signal.iirfilter(1, Wn=1, fs=165, btype="low")
    return signal.filtfilt(b, a, x)

def stim_load(filename):
    return pd.read_csv(filename + '_cluttered.csv')

neurons = ['STMD_R', 'STMD_L', 'LPTC_R', 'LPTC_L', 'LPTC_HR']

stims = []
for stim in stimuli:
    df = stim_load(stim)
    filtered_neurons = []
    for n in neurons:
        filtered_neurons.append(filtering(df[n].to_numpy()))
    stims.append(np.vstack(filtered_neurons))
    
stims = np.stack(stims) # (stimuli, neurons, timesteps)

sdfs = pickle.load(open('sdfs_cluttered.pkl', 'rb'))

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

fig, axes = plt.subplots(5, figsize=(6, 10), dpi=500, sharex=True, sharey=True)
for j, ax in enumerate(axes):
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
            sim = model_D([85, 1.1, 0.4], [STMD_R, LPTC_R, LPTC_L])
        elif j == 1:
            title = 'E'
            sim = model_E([44, 0.014], [STMD_R, LPTC_HR])
        elif j == 2:
            title = 'F'
            sim = model_F([17, 0.01, 1.1], [STMD_R, LPTC_R, LPTC_L])
        elif j == 3:
            title = 'G'
            sim = model_G([44, 1.4, 0], [STMD_R, STMD_L, LPTC_R])
        elif j == 4:
            title = 'H'
            sim = model_H([46, 0.05, 0.025], [STMD_R, LPTC_R, LPTC_L])
            
        stimuli[i] = stimuli[i].replace('_', ' ')
            
        axes[j].set_title(title)
        axes[j].plot(sim, label=stimuli[i])
axes[-1].set_xlabel('Time [ms]')
fig.supylabel('Activation [a.u.]')
plt.legend()
        
            
        
            
    
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