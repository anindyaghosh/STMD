import matplotlib.pyplot as plt
import numpy as np

"""
STMD modelling for simple homogeneous stimulus translating to the right
"""

# =============================================================================
# X-Y map of object moving
# =============================================================================

t_sim = 100 # timesteps
v = 20 # pix/timestep
offset = 0
num_neurons = 50
height = 2 # pix

vf = np.zeros((num_neurons, t_sim))
ylims = (vf.shape[0]-height) // 2 + np.arange(height) + offset # offset in y_direction
start = [ylims, 0]

def translation(t, v, x_init):
    return v * t + x_init

for t in range(t_sim):
    dist = translation(t, v, x_init=start[1])
    if dist < vf.shape[1]:
        vf[start[0], dist] = 10
    else:
        break
        
# =============================================================================
# Neuron models - LIF
# =============================================================================

# F-I curve
# k = slope, x0 = midpoint
pars_F = {'b': 1.0, 'k': 1.2, 'x0': 5.0}
pars_dyn = {'w_E': 0.6, 'w_I': 0.2, 'dt': 1.0, 'tau': 2.0, 'r_init': 0.0}

r = np.zeros(vf.shape)
r[0] = pars_dyn['r_init']

dt, tau = pars_dyn['dt'], pars_dyn['tau']
w_E, w_I = pars_dyn['w_E'], pars_dyn['w_I']

class neuron_dynamics:
    def __init__(self, **pars_FI):
        self.b = pars_FI['b']
        self.k = pars_FI['k']
        self.x0 = pars_FI['x0']
    
    def F(self, b, k, I, x0):
        return b / (1 + np.exp(-k * (I-x0)))

    def simulate(self, w, neuron_pop):
        for i in range(t_sim-1):
            I_ext = neuron_pop[:,i]
            dr = dt / tau * (-r[:,i] + self.F(self.b, self.k, w * r[:,i] + I_ext, self.x0))
            r[:,i+1] = r[:,i] + dr
        
        # fig, ax = plt.subplots()
        # ax.plot(range(t_sim), r[num_neurons // 2,:])
            
        return r

x = neuron_dynamics(**pars_F)

tol = 0.01
rE = x.simulate(10, vf) # photoreceptor (P), P-E and P-I synaptic weights = 1
rE[rE < tol] = 0

rI = rE.copy()

# Lateral inhibition
I_STMD = np.zeros(rI.shape)

w = {'a': -0.33, 'b': -0.22}
c_i = np.array(['b', 'a', 'a', 'b'])
w_spatial_mat = np.array([-2, -1, 1, 2])

w_i = np.asarray([w[i] for i in c_i])

for col in range(t_sim-2):
    for row in range(rI.shape[0]):
        if row < rI.shape[0] - (len(w_spatial_mat) // 2):
            I_STMD[row + w_spatial_mat, col+2] += w_i * rI[row, col]

# Inverse to get current and then plug into LIF to get firing rate or rate fine as is?
            
STMD = pars_dyn['w_E'] * rE + pars_dyn['w_I'] * I_STMD
print(STMD.max())

fig, ax = plt.subplots()
ax.plot(np.arange(10, step=0.1), x.F(pars_F['b'], pars_F['k'], np.arange(10, step=0.1), pars_F['x0']))
plt.xlabel('I [a.u.]')
plt.ylabel('Firing rate [Hz]')