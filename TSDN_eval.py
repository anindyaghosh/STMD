import matplotlib.pyplot as plt
import numpy as np

with open('d-STMD_and_EMD.txt', 'r') as f:
    stim_vals = []
    values = []
    lines = f.read().splitlines()
    for line in lines:
        stim, vals = line.split(';')
        stim_vals.append(stim)
        temp_vals = [eval(v) for v in vals.split(',')]
        values.append(np.array(temp_vals))
    values = np.vstack(values)
        
normalise = np.hstack((values[:,:2]/np.max(values[:,:2]), values[:,2:]/np.max(values[:,2:])))
norm_vals = np.vstack((np.clip(normalise[:,0] - normalise[:,2], 0, None), np.clip(normalise[:,1] - normalise[:,3], 0, None))).T
values = np.concatenate((values, norm_vals), axis=1)
values = np.c_[values, np.sum(norm_vals, axis=1)]

def plots(labels, offset):
    fig, axes = plt.subplots(figsize=(10,6))
    x = np.arange(3)
    w = 0.4
    plt.xticks(x + w/3, stim_vals)
    
    num = len(labels)
    
    if any('STMD' in l for l in labels):
        axes2 = axes.twinx()
        for i in range(num):
            if 'STMD' in labels[i]:
                axes.bar(x+w*i/num, values[:,i+offset], width=w/num, align='center', label=labels[i], edgecolor='black')
            else:
                axes2.bar(x+w*i/num, values[:,i+offset], width=w/num, align='center', hatch='/', label=labels[i], edgecolor='black')
        axes.set_xlabel('Relative motion')
        axes.set_ylabel('ESTMD activity [a.u.]')
        axes2.set_ylabel('EMD activity [a.u.]')
        
        lns, labs = axes.get_legend_handles_labels()
        lns2, labs2 = axes2.get_legend_handles_labels()
        axes2.legend(lns + lns2, labs + labs2)
        
    else:
        for i in range(num):
            axes.bar(x+w*i/num, values[:,i+offset], width=w/num, align='center', label=labels[i], edgecolor='black')
        axes.set_xlabel('Relative motion')
        axes.set_ylabel('TSDN activity [normalised a.u.]')
        axes.legend()
    
plots(['dSTMD right', 'dSTMD left', 'EMD right', 'EMD left'], 0)
plots(['TSDN right', 'TSDN left', 'TSDN total'], 4)