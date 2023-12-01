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

args = parser.parse_args(['--model_type', 'E', '--stimulus_type', 'starfield'])
# args = parser.parse_args()

stimuli = ['alone', 'stationary', 'background_right', 'background_left']

def filtering(x):
    b, a = signal.iirfilter(1, Wn=1, fs=165, btype="low")
    return signal.filtfilt(b, a, x)

def stim_load(filename):
    return pd.read_csv(f'{filename}_{args.stimulus_type}' + '.csv')

neurons = ['STMD_R', 'STMD_L', 'LPTC_R', 'LPTC_L', 'LPTC_HR']

# In frames
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
            
            # normalised RMSE
        err += (np.sum((TSDN - TSDN_ephys)**2) / len(TSDN_ephys))
    return np.sqrt(err / stims.shape[-1])

if args.model_type == 'D':
    arguments = []
    for STMD_R in np.arange(2000, 8000, 50):
        for LPTC_R in np.arange(5, 50, 1):
            for LPTC_L in np.arange(0, 250, 1):
                # for shift in np.arange(0, 1000, 25):
                #     for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, LPTC_R, LPTC_L])
                    
elif args.model_type == 'E':
    arguments = []
    for STMD_R in np.arange(150, 300, 0.5):
        for LPTC_HR in np.arange(0, 0.025, 0.001):
            # for shift in np.arange(0, 1000, 25):
            #     for LPTC_shift in np.arange(0, 1000, 25):
            arguments.append([STMD_R, LPTC_HR])
            
elif args.model_type == 'F':
    arguments = []
    for STMD_R in np.arange(150, 250, 0.5):
        for LPTC_R in np.arange(0, 0.05, 0.001):
            for LPTC_L in np.arange(0, 25, 0.5):
                # for shift in np.arange(0, 1000, 25):
                #     for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, LPTC_R, LPTC_L])
                    
elif args.model_type == 'G':
    arguments = []
    for STMD_R in np.arange(200, 400, 1):
        for STMD_L in np.arange(200, 250, 1):
            for LPTC_R in np.arange(0, 0.001, 0.0001):
                # for shift in np.arange(0, 1000, 25):
                    # for LPTC_shift in np.arange(0, 1000, 25):
                arguments.append([STMD_R, STMD_L, LPTC_R])
                
elif args.model_type == 'H':
    arguments = []
    for STMD_R in np.arange(200, 300, 1):
        for LPTC_R in np.arange(0, 0.01, 0.0005):
            for LPTC_L in np.arange(0, 0.025, 0.0005):
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