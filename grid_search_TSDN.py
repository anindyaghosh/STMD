import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy import signal
from tqdm import tqdm

# import TSDN_data_analysis

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)
parser.add_argument('--stimulus_type', type=str)

args = parser.parse_args(['--model_type', 'E', '--stimulus_type', 'sinusoidal'])

stimuli = ['alone', 'stationary', 'background_right', 'background_left']

def filtering(x):
    b, a = signal.iirfilter(1, Wn=1, fs=165, btype="low")
    return signal.filtfilt(b, a, x)

def stim_load(filename):
    return pd.read_csv(f'{filename}_{args.stimulus_type}' + '.csv')

neurons = ['STMD_R', 'STMD_L', 'LPTC_R', 'LPTC_L', 'LPTC_HR']

stims = []
for stim in stimuli:
    df = stim_load(stim)
    filtered_neurons = []
    for n in neurons:
        filtered_neurons.append(filtering(df[n].to_numpy()))
    stims.append(np.vstack(filtered_neurons))
    
stims = np.stack(stims) # (stimuli, neurons, timesteps)

# sdfs = TSDN_data_analysis.averages_all
sdfs = pickle.load(open(f'sdfs_{args.stimulus_type}.pkl', 'rb'))

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

def model(params):
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
        
        if args.model_type == 'D':
            TSDN = model_D(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
        elif args.model_type == 'E':
            TSDN = model_E(params, [STMD_R, LPTC_HR])[:len(TSDN_ephys)]
        elif args.model_type == 'F':
            TSDN = model_F(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
        elif args.model_type == 'G':
            TSDN = model_G(params, [STMD_R, STMD_L, LPTC_R])[:len(TSDN_ephys)]
        elif args.model_type == 'H':
            TSDN = model_H(params, [STMD_R, LPTC_R, LPTC_L])[:len(TSDN_ephys)]
        else:
            print('Unknown model type')
            
        err += np.sum((TSDN - TSDN_ephys)**2) / stims.shape[-1]
    return np.sqrt(err)

if args.model_type == 'D':
    arguments = []
    for STMD_R in np.arange(50, 100, 1):
        for LPTC_R in np.arange(0.3, 1.25, 0.05):
            for LPTC_L in np.arange(0, 1, 0.05):
                # for shift in np.arange(0, 1000, 25):
                #     for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, LPTC_R, LPTC_L])
                    
elif args.model_type == 'E':
    arguments = []
    for STMD_R in np.arange(30, 60, 1):
        for LPTC_HR in np.arange(0, 0.1, 0.001):
            # for shift in np.arange(0, 1000, 25):
            #     for LPTC_shift in np.arange(0, 1000, 25):
            arguments.append([STMD_R, LPTC_HR])
            
elif args.model_type == 'F':
    arguments = []
    for STMD_R in np.arange(0, 70, 1):
        for LPTC_R in [0, 0.001]:
            for LPTC_L in np.arange(0, 3.5, 0.05):
                # for shift in np.arange(0, 1000, 25):
                #     for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, LPTC_R, LPTC_L])
                    
elif args.model_type == 'G':
    arguments = []
    for STMD_R in np.arange(20, 70, 1):
        for STMD_L in np.arange(0, 15, 5):
            for LPTC_R in np.arange(0, 0.4, 0.05):
                # for shift in np.arange(0, 1000, 25):
                    # for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, STMD_L, LPTC_R])
                
elif args.model_type == 'H':
    arguments = []
    for STMD_R in np.arange(20, 80, 1):
        for LPTC_R in np.arange(0, 0.08, 0.005):
            for LPTC_L in np.arange(0, 0.05, 0.005):
                # for shift in np.arange(0, 1000, 25):
                    # for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, LPTC_R, LPTC_L])

with tqdm(total=len(arguments)) as pbar:
    model_output = []
    for params in arguments:
        model_output.append((params, model(params)))
        pbar.update(1)

save_dir = os.path.join(os.getcwd(), 'grid_search_saves')

os.makedirs(save_dir, exist_ok=True)

df = pd.DataFrame(model_output)
df.to_csv(os.path.join(save_dir, f'{args.model_type}.csv'))