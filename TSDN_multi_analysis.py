import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.io
from scipy import signal, stats
import seaborn as sns

sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
sns.set_palette("deep")

stim_type = 'starfield' # cluttered, sinusoidal or starfield

root = os.getcwd()
data_folder = os.path.join(root, 'extractions', f'{stim_type}')
files = os.listdir(data_folder)

labels = ['Alone (no background)', 'Stationary background', 'Same direction', 'Opposite direction']

def culling_stims(var):
    sample_per_frame = params['total_samples'] / params['total_frames']
    to_be_removed = int((80-pre_remaining) * sample_per_frame)
    stim_time = int(160 * sample_per_frame)
    post_stim = int(post_remaining * sample_per_frame)
    
    return var[:,to_be_removed:to_be_removed+stim_time+post_stim]
    
def spike_viz(var):
    for i in var:
        fig, axis = plt.subplots()
        axis.plot(time_axis, i)

# Convert to continuous-time domain
from scipy.integrate import simpson #  to find area under curve

class spike_evaluation():
    def ddf(self, x, sigma):
        delta = 1 / (self.sigma * np.sqrt(2*np.pi)) * np.exp(-(x**2) / (2 * self.sigma**2))
        area = simpson(delta[:,0], dx=x[0]-x[1]) # should be 1 or else normalise
        return delta / area
    
    def superposition(self, x, FIR):
        
        sum_FIR = np.zeros(self.time + 1)
        d = x.ravel()
        f = lambda s: np.unravel_index(s.index, x.shape)
        inds = pd.Series(d).groupby(d).apply(f)
        
        for i, t in enumerate(inds.index):
            if t <= self.time:
                t = int(t)
                sum_FIR[t] = np.sum(FIR[inds.iloc[i]])
        
        t_sum = np.arange(self.time + 1)
        
        return t_sum, np.array(sum_FIR)
    
    def spike_density_function(self, t_i, r, sigma):
        self.sigma = sigma
        
        t_i = (time_axis[np.nonzero(t_i)[0]]).astype(int)
        self.time = time_axis[-1].astype(int)
        # fig, axis = plt.subplots()
        sdf = []
        for i, t in enumerate([t_i]):
            x1, x2 = (j + t for j in (r, -r))
            x = np.linspace(x1, x2, 2*r+1)
            FIR = self.ddf(x - t, self.sigma) # finite energy impulse response
            
            # Clipping Gaussians to time of collision
            # FIR[np.where(x > len(self.t_vals))] = 0
            
            # axis.plot(x, FIR)
            if FIR.size > 0:
                t_sum, sum_FIR = self.superposition(x, FIR)
            else:
                t_sum, sum_FIR = np.array([]), np.array([])
    
            sdf.append((t_sum, sum_FIR))
                
        return np.squeeze(sdf)
    
def match_check(vars):
    sdf_all = []
    for v, var in enumerate(vars):
        sdfs, kurs = [], []
        empty_count = []
        for spk, spike_train in enumerate(var):
            if np.squeeze(np.nonzero(spike_train)).size:
                x = spike_evaluation()
                sdf = x.spike_density_function(spike_train, r=100, sigma=20)
                sdfs.append(sdf)
                
                # # Test for heavy-tailed distributions -- assuming normal distribution
                # kur = stats.kurtosis(sdf[1])
                # kurs.append(kur)
                
                # fig, axis = plt.subplots()
                # axis.plot(sdf[0], sdf[1] / np.max(sdf[1]))
            else:
                sdf = sdf.copy()
                sdf[1,:] = np.zeros((1, sdf.shape[1]))
                sdfs.append(sdf)
                # print(f"Stim: {v}, no: {spk} is empty")
                empty_count.append(spk)
                
        # var_size = var.shape[0] - len(empty_count) # in case I take out if statement block on Ln 135
        # results = np.zeros((var_size, var_size))
        # for i, sdf_row in enumerate(sdfs):
        #     for j, sdf_col in enumerate(sdfs):
        #         results[i,j] = stats.ks_2samp(sdf_row[1], sdf_col[1]).pvalue
                
        # results[results < 0.01] = 0 # 99% confidence level. Null hypothesis is that two samples were drawn from the same distribution.
        # hist = np.bincount(np.nonzero(results)[0])
        
        # fig, axes = plt.subplots(dpi=500)
        # if len(empty_count) > 0:
        #     for c in empty_count:
        #         hist = np.insert(hist, c, 0)
        #     var_size += len(empty_count)
        # axes.bar(np.arange(var_size), (hist-1).clip(min=0))
        # axes.set_xticks(range(var_size), np.arange(var_size).astype(str))
        # axes.axhline(y=np.max(hist-1)/2, color='k', linestyle='--')
        
        sdf_all.append(sdfs)
        
    return sdf_all

def moving_average(vars):
    avgs_all = []
    for var in vars:
        avgs = []
        for v in var:
            avgs.append(np.convolve(v, np.ones(sliding_window_length)/bin_size, mode='full'))
            
        avgs_all.append(np.mean(np.vstack(avgs), axis=0))
    return avgs_all

def ewma(vars):
    avgs_all = []
    for var in vars:
        avgs = []
        for v in var:
            df = pd.DataFrame(v)
            rolling_average = df.iloc[:,0].ewm(alpha=np.exp(-1/20)).mean()
            avgs.append(rolling_average)
            
        avgs_all.append(np.vstack(avgs))
    return avgs_all

def dirac_lpf(vars):
    avgs_all = []
    for var in vars:
        avgs_stimwise = []
        for spike_train in var:
            if np.squeeze(np.nonzero(spike_train)).size:
                x = spike_evaluation()
                sdf = x.spike_density_function(spike_train, r=100, sigma=20)[1]
            else:
                sdf = np.zeros((sdf.shape))
            avgs_stimwise.append(sdf)
        avgs_all.append(avgs_stimwise)
        
    return avgs_all

def axis_info(axis):
    axis.set_xlabel('Time [ms]', labelpad=80)
    
    axis.set_xlim([0, total_time]) # Total time
    axis.set_ylim([0, None])
    # axis.axvline(x=730, color='blue', linestyle='dashed')
    # axis.axvline(x=745, color='red', linestyle='dashed')
    
    trans = axis.get_xaxis_transform()
    axis.plot([pre_stim, pre_stim+stim_time], [-0.15,-0.15], color="blue", transform=trans, clip_on=False)
    axis.annotate('Background on', xy=(total_time-post_stim, -0.2), xycoords=trans, ha="right", va="top", color='blue')
    
    axis.plot([pre_stim + stim_time/2, pre_stim+stim_time], [-0.3,-0.3], color="red", transform=trans, clip_on=False)
    axis.annotate('Target on', xy=(total_time-post_stim, -0.35), xycoords=trans, ha="right", va="top", color='red')
        
    plt.legend()

def moving_average_viz(moving_avgs):
    fig, axis = plt.subplots(dpi=500)
    for idx, i in enumerate(moving_avgs):
        axis.plot(time_axis_bins, i, label=labels[idx])
        axis.set_ylabel('Spike response frequency [$ms^{-1}$]')
        
    axis_info(axis)
    
    fig, axis = plt.subplots(dpi=500, sharex=True, sharey=True)
    ewma_averages = dirac_lpf([alone, stationary, same_direction, opposite_direction])
    for idx, stim in enumerate(ewma_averages):
        avg_values = np.mean(np.vstack(stim), axis=0)
        axis.plot(avg_values, label=labels[idx])
        axis.set_ylabel('Spike density function [$ms^{-1}$]')
        
    axis_info(axis)
    
def spike_train_viz(vars, mode='spikes'):
    if mode == 'spikes':
        fig, axes = plt.subplots(len(vars), figsize=(10, 24), dpi=500, sharex=True)
        for i, ax in enumerate(axes.flatten()):
            spikes = []
            for v in vars[i]:
                spikes.append((time_axis[np.nonzero(v)[0]]).astype(int))
            ax.eventplot(spikes, linelengths=0.5)
            ax.set_yticks(np.arange(len(spikes)))
            ax.set_title(labels[i])
            
        fig.supylabel('Trial indices [#]')
        
    elif mode == 'ewma':
        fig, axes = plt.subplots(len(vars), figsize=(5, 12), dpi=500, sharex=True, sharey=True)
        ewma_averages = ewma(vars)
        for i, ax in enumerate(axes.flatten()):
            ewma_downsampled = []
            for train in ewma_averages[i]:
                ewma_downsampled.append(np.abs(signal.resample(train, int(total_time))))
            avg_values = np.mean(np.vstack(ewma_downsampled), axis=0)
            ax.plot(np.arange(int(total_time)), avg_values)
            ax.set_title(labels[i])
            
        fig.supylabel('Spike frequency [$ms^{-1}$]')
        
    trans = axes[-1].get_xaxis_transform()
    axes[-1].plot([pre_stim, pre_stim+stim_time], [-0.2,-0.2], color="blue", transform=trans, clip_on=False)
    axes[-1].annotate('Background on', xy=(total_time-post_stim, -0.25), xycoords=trans, ha="right", va="top", color='blue')
    
    axes[-1].plot([pre_stim + stim_time/2, pre_stim+stim_time], [-0.4,-0.4], color="red", transform=trans, clip_on=False)
    axes[-1].annotate('Target on', xy=(total_time-post_stim, -0.45), xycoords=trans, ha="right", va="top", color='red')
    
    plt.xlabel('Time [ms]', labelpad=80)
    plt.xlim([0, total_time]) # Total time
    
    plt.tight_layout()

all_sdfs = []
for file in files:
    print(file)
    stimulus = scipy.io.loadmat(os.path.join(data_folder, file))
    output = stimulus['output']
    params = {param: np.squeeze(stimulus[param]) for param in ['frame_rate', 
                                                                'post_stim',
                                                                'pre_stim', 
                                                                'scan_duration', 
                                                                'sf', # spike frequency in samples/second
                                                                'total_frames', 
                                                                'total_samples', 
                                                                'total_time',
                                                                'Target_Direction']}
    
    # In frames
    pre_remaining = 5
    post_remaining = 5
    
    params.update({'pre_stim': (pre_remaining+80)/params['frame_rate'], 'post_stim': post_remaining/params['frame_rate']})
    params.update({'total_time': params['pre_stim'] + params['scan_duration'] + params['post_stim']})
    
    time_axis = np.linspace(0, params['total_time'], round(params['total_time'] * params['sf'])) * 1000 # ms

    # Case values for target moving to right
    alone = []
    stationary = [] # target stationary
    same_direction = []
    opposite_direction = []
    
    if stim_type == 'cluttered':
        # TODO: look at cluttered
        pass
        # for i in range(output.shape[0]):
        #     if output[i,0] == 0:
        #         alone.append(output[i,:])
        #     elif output[i,0] == 160 and output[i,1] == 0:
        #         stationary.append(output[i,:])
        #     elif output[i,0] == 160 and output[i,1] == 900:
        #         same_direction.append(output[i,:])
        #     elif output[i,0] == 160 and output[i,1] == -900:
        #         opposite_direction.append(output[i,:])
    elif stim_type == 'sinusoidal':
        # TODO: look at sinusoidal
        pass
        # for i in range(output.shape[0]):
        #     if output[i,0] == 0:
        #         alone.append(output[i,:])
        #     elif output[i,0] > 159 and output[i,1] == 0:
        #         stationary.append(output[i,:])
        #     elif output[i,0] > 159 and output[i,1] == 5 and output[i,2] == 0:
        #         same_direction.append(output[i,:])
        #     elif output[i,0] > 159 and output[i,1] == 5 and output[i,2] == 180:
        #         opposite_direction.append(output[i,:])
    elif stim_type == 'starfield':
        for i in range(output.shape[0]):
            if output[i,0] == 0:
                alone.append(output[i,:])
            elif output[i,0] == 1 and output[i,1] == 0:
                stationary.append(output[i,:])
            else:
                if params['Target_Direction'][0] == 0:
                    if output[i,0] == 1 and output[i,1] == 50:
                        same_direction.append(output[i,:])
                    elif output[i,0] == 1 and output[i,1] == -50:
                        opposite_direction.append(output[i,:])
                else:
                    if output[i,0] == 1 and output[i,1] == -50:
                        same_direction.append(output[i,:])
                    elif output[i,0] == 1 and output[i,1] == 50:
                        opposite_direction.append(output[i,:])

    alone = culling_stims(np.vstack(alone)[:,2:])
    stationary = culling_stims(np.vstack(stationary)[:,2:])
    same_direction = culling_stims(np.vstack(same_direction)[:,2:])
    opposite_direction = culling_stims(np.vstack(opposite_direction)[:,2:])
    
    bin_size = 50 # ms
    
    # spike_viz([alone[0]])

    # sdfs = match_check([alone, stationary, same_direction, opposite_direction])
    
    # alone = alone[[0, 1, 2, 4, 5, 6, 9, 10],:]
    # stationary = stationary[[1, 3, 4, 6, 7, 8, 9],:]
    # same_direction = same_direction[[0, 2, 5, 6, 9, 10],:]
    # opposite_direction = opposite_direction[[1, 2, 4, 5],:]
    
    sliding_window_length = int(bin_size * params['sf'] / 1000)
    
    pre_stim = pre_remaining/165 * 1000
    stim_time = 160/165 * 1000
    post_stim = post_remaining/165 * 1000
    
    total_time = pre_stim + stim_time + post_stim

    moving_avgs = moving_average([alone, stationary, same_direction, opposite_direction])
    ewma_averages = ewma([alone, stationary, same_direction, opposite_direction])
    
    experiment_name = os.path.splitext(file)[0]

    if not os.path.isfile(f'sdfs_{stim_type}_{experiment_name}.pkl'):
        averages_all = dirac_lpf([alone, stationary, same_direction, opposite_direction])
        pickle.dump(averages_all, open(f'sdfs_{stim_type}_{experiment_name}.pkl', 'wb'))
    else:
        averages_all = pickle.load(open(f'sdfs_{stim_type}_{experiment_name}.pkl', 'rb'))
    all_sdfs.append(averages_all)
    
    time_axis_bins = np.linspace(0, time_axis[-1], len(moving_avgs[0])).astype(int)
    
    # moving_average_viz(moving_avgs)
    # spike_train_viz([alone, stationary, same_direction, opposite_direction], mode='spikes')
    
fig, axes = plt.subplots(4, 1, figsize=(8, 12), dpi=500, sharex=True, sharey=True)
all_experiments = [[] for i in range(4)]
for experiment in all_sdfs:
    for j, exp in enumerate(experiment):
        average_experiments = np.mean(np.vstack(exp), axis=0)
        all_experiments[j].append((np.argmax(average_experiments), 
                                   np.max(average_experiments), 
                                   average_experiments))
        
        axes[j].plot(average_experiments)
        axes[j].set_title(labels[j])
fig.supylabel('Spike density function [$ms^{-1}$]')
axis_info(axes[-1])
axes[-1].get_legend().remove()

rf_stmd_centres = np.load('rf_freq.npy')

fig, axis = plt.subplots(figsize=(9, 5), dpi=500)
for i, stim in enumerate(all_experiments):
    cumulative_trials = 0
    stim_arr = np.asarray(stim, dtype=object)
    
    # Shift all trials so position of maximum lines up
    shifts = (stim_arr[:,0] - stim_arr[np.argmax(stim_arr[:,1]),0]).astype(int)
    shifts = -np.min(-shifts) - shifts
    
    # Find cutoff
    cutoff = min([shifts[i] + len(trial) for i, trial in enumerate(stim_arr[:,2])])
    for s, trial in enumerate(stim_arr):
        shifted_trial = np.pad(trial[2], (shifts[s], 0))
        # Shows plots of all peaks for stimulus type
        # if i == 0:
        #     axis.plot(shifted_trial)
        cumulative_trials += shifted_trial[:cutoff]
        
    stim_mean = cumulative_trials / s
    axis.plot(stim_mean, label=labels[i])
fig.supylabel('Spike density function [$ms^{-1}$]')
# axis.set_xlabel('Time [ms]')
axis_info(axis)
plt.legend()

def trendline(subset):
    z = np.polyfit(subset[:,0], subset[:,1], 1)
    return np.poly1d(z)

fig, axes = plt.subplots(4, 1, figsize=(7,10), dpi=500, sharex=True, sharey=True)
for i, stim in enumerate(all_experiments):
    stim_arr = np.asarray(stim, dtype=object)
    axes[i].set_title(labels[i])
    peaks = []
    for j in range(len(rf_stmd_centres)):
        peaks.append(np.max(stim_arr[j,2]))
        axes[i].scatter(rf_stmd_centres[j], np.max(stim_arr[j,2]), color='black')
        
    # Fit trendline till STMD centres <= 11 and >= 13 using subset mask
    stack_arr = np.vstack([rf_stmd_centres, peaks]).T
    stack_arr = stack_arr[stack_arr[:, 0].argsort()]
    
    subset_low = stack_arr[stack_arr[:,0] <= 11]
    subset_high = stack_arr[stack_arr[:,0] >= 13]
    
    p_low = trendline(subset_low)
    p_high = trendline(subset_high)
    
    axes[i].plot(subset_low[:,0], p_low(subset_low[:,1]), 'r--')
    axes[i].plot(subset_high[:,0], p_high(subset_high[:,1]), 'b--')
    
axes[-1].set_xlabel('Number of STMD centres')
fig.supylabel('Spike response frequency [$ms^{-1}$]')