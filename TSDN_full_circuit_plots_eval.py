import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# from helper_functions import set_size

images = ["clutter image.png", "all target image.png"]

def offset_image(x, y, image, ax):
    img = plt.imread(image)
    im = OffsetImage(img, zoom=0.3)
    im.image.axes = ax
    ab = AnnotationBbox(im, (x, y), xycoords='axes fraction', frameon=False, pad=0)
    ax.add_artist(ab)

with open('TSDN_full_circuit_plots.txt', 'r') as f:
    neurons = {}
    for line in f:
        motion, results = line.split('~')
        neurons[motion] = {}
        for i in results.split(';'):
            key, values = i.split(':')
            neurons[motion][key] = eval(values)

direction = 'right'
neurons_ = ['STMD', 'LPTC', 'wSTMD', 'LPTC_baseline']

arr = []
for n in neurons.values():
    arr.append([n[i] for i in neurons_])
    
arr = np.stack(np.asarray(arr)).T
index = 1 if direction == 'right' else 0

# Right in zeroth column, then left in first column
STMD = arr[1-index,0,:]
LPTC = arr[:,1,:].T
wSTMD = arr[index,2,:]
LPTC_baseline = (arr[0,3,:].T).clip(min=0)

def normalise(neuron):
    noise = neuron[:-3,...]
    target = neuron[-3:,...]
    
    noise /= np.max(noise)
    target /= np.max(target)
    
    return np.concatenate((noise, target))

STMD = normalise(STMD)
LPTC = normalise(LPTC)
wSTMD = normalise(wSTMD)
LPTC_baseline = normalise(LPTC_baseline)

LPTC_inhib = (3 * LPTC[:,1-index] - LPTC[:,index]).clip(min=0)

configs = [STMD - LPTC[:,1-index], STMD - LPTC_inhib, STMD - LPTC[:,1-index] + wSTMD, STMD - LPTC_inhib + wSTMD, STMD - LPTC_baseline]

for c, config in enumerate(configs):
    config = config.clip(min=0)
    
    width = 469.75502
    fig, axes = plt.subplots(figsize=(8,5), dpi=200)
    x = np.arange(len(neurons.keys()))
    w = 0.5
    labels = []
    mapping_table = str.maketrans({' ':'\n', '-':'-\n', '#':''})
    for i in neurons.keys():
        labels.append(i.split('_noise')[0].translate(mapping_table))
    plt.xticks(x, labels)
    ax2 = axes.twinx()
    
    TSDN = axes.bar(x[:-3], config[:-3], width=w, align='center', edgecolor='black')
    target_bg = ax2.bar(x[-3:], config[-3:], width=w, align='center', color='#ff7f0e', edgecolor='black')
    axes.set_xlabel('Relative motion')
    axes.set_ylabel('TSDN activity [a.u.]')
    ax2.set_ylabel('TSDN activity [a.u.]')
    if c <= 0:
        loc = 'upper center'
    else:
        loc = 'best'
    plt.legend([TSDN, target_bg], ['clutter', 'target background'], loc=loc)
    offset_image(0.11, -0.25, images[0], axes)
    offset_image(0.72, -0.25, images[1], axes)
    
with open('LPTC_velocity_tuning.txt', 'r') as f:
    LPTC_results = []
    for line in f:
        LPTC_results.append(eval(line))
        
LPTC_results = np.asarray(LPTC_results)

fig, axes = plt.subplots(dpi=500)
axes.plot(LPTC_results[:,0], (LPTC_results[:,1]).clip(min=0), label='LPTC activation')
axes.plot(LPTC_results[:,0], LPTC_results[:,2], label='LPTC spontaneous activity')
axes.set_xlabel('Background velocity [$\degree$/s]')
axes.set_ylabel('LPTC activation [a.u.]')
plt.legend()

"""
0. STMD_right - LPTC_right -- (Standard model)
1. STMD_right + ReLU(LPTC_right - LPTC_left) -- (Option 1a)
2. STMD_right - LPTC_right + wSTMD -- (Option 2b)
3. STMD_right + ReLU(LPTC_right - LPTC_left) + wSTMD -- (Hybrid of options 1a and 2b)
4. STMD_right - independent LPTC -- (Option 2b)
"""