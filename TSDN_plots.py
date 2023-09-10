import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filenames = ['ESTMD_EMD', 'PoC_EMD']

def read_file(filename):
    with open(filename + '.txt', 'r') as f:
        lines = f.read().splitlines()
        return [eval(line) for line in lines]
        
vals = []
for file in filenames:
    vals.append(read_file(file))

# EMD activity
# EMD_vals = np.asarray(vals[:4]).T
# labels = ['right', 'down']
# fig, axes = plt.subplots()
# for i, v in enumerate(EMD_vals):
#     axes.plot([0, 90, 180, 270], v, label=labels[i])
# axes.set_xlabel('Relative motion [degrees]')
# axes.set_ylabel('EMD Activity [a.u.]')
# axes.legend()
# plt.xticks([0, 90, 180, 270], [0, 90, 180, 270])

# ESTMD activity
# ESTMD_vals = np.asarray(vals[4:-1])
# fig, axes = plt.subplots()
# axes.plot([0, 90, 180, 270], ESTMD_vals)
# axes.set_xlabel('Relative motion [degrees]')
# axes.set_ylabel('ESTMD Activity [a.u.]')
# plt.xticks([0, 90, 180, 270], [0, 90, 180, 270])

proof = vals[-1]
ESTMD_EMD_vals = vals[0]

proof = np.asarray(proof)
ESTMD_EMD_vals = np.asarray(ESTMD_EMD_vals)

# Normalised activity
# ESTMD_EMD_vals[:,0] /= np.max(ESTMD_EMD_vals[:,0])
# ESTMD_EMD_vals[:,1:] /= np.max(ESTMD_EMD_vals[:,1:])

# TSDN_vals = ESTMD_vals - EMD_vals

def plots(**kwargs):
    if kwargs['stim'] == 'proof':
        num = 4
        label = [0, 90, 180, 270]
        arr = proof
    else:
        num = 7
        label = ['Grey', 'Stationary', 'No target', 0, 90, 180, 270]
        arr = ESTMD_EMD_vals
    
    vals_pd = pd.DataFrame(arr, columns = ['ESTMD (left axis)',
                                           'EMD-right (right axis)','EMD-down (right axis)'])
    
    fig, axes = plt.subplots(figsize=(9,6), dpi=500)
    x = np.arange(num)
    w = 0.3
    plt.xticks(x + w/3, label)
    ESTMD = axes.bar(x, vals_pd['ESTMD (left axis)'], width=w/3, color='b', align='center')
    axes2 = axes.twinx()
    EMD_right = axes2.bar(x + w/3, vals_pd['EMD-right (right axis)'], width=w/3, color='g',align='center')
    EMD_down = axes2.bar(x + 2*w/3, vals_pd['EMD-down (right axis)'], width=w/3, color='r', align='center')
    axes.set_xlabel('Relative motion [degrees]')
    axes.set_ylabel('ESTMD activity [a.u.]')
    axes2.set_ylabel('EMD activity [a.u.]')
    plt.legend([ESTMD, EMD_right, EMD_down], ['ESTMD','EMD-right','EMD-down'], loc='upper center')
    
plots(stim='proof')