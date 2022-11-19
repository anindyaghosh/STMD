import numpy as np

# =============================================================================
# X-Y map of object moving
# =============================================================================
t_sim = 100 # timesteps
v = 5 # pix/timestep
vf = np.zeros((100, 100))

height = 5 # pix
vf = np.zeros((100, 100))

height = 5
ylims = (vf.shape[0]-height) // 2 + np.arange(height)
start = [ylims, 0]

def translation(t, v, x_init):
    return v * t + x_init

x = []
for t in range(t_sim):
    dist = translation(t, v, x_init=start[1])
    if dist < vf.shape[1]:
        x.append(dist)
        
        vf[start[0], dist] = 1
        
# =============================================================================
# Neuron models - LIF
# =============================================================================
