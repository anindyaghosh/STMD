import itertools
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_starfield = [["180501", (1301,491), 1164, 1422, 378, 611],
                  ["180508", (1191,502), 1086, 1280, 407, 596],
                  ["180620", (1208,407), 1112, 1293, 291, 538],
                  ["180628", (1347,397), 1267, 1422, 262, 502],
                  ["180801", (1229,497), 1138, 1345, 364, 611],
                  ["181023", (1331,513), 1241, 1422, 436, 582],
                  ["181121", (1213,458), 1138, 1293, 364, 553],
                  ["181129", (1175,393), 1060, 1293, 276, 524],
                  ["190219", (1211,446), 1138, 1293, 378, 509],
                  ["190501_1", (1216,399), 1112, 1319, 320, 495],
                  ["190501_3", (1087,477), 931, 1267, 291, 640],
                  ["190530", (1301,344), 1138, 1422, 160, 524],
                  ["190726", (1166, 321), 983, 1371, 218, 437],
                  ["190806", (1205,424), 1112, 1293, 349, 495],
                  ["190807", (1168,377), 1034, 1293, 204, 524],
                  ["191101", (1205,477), 1112, 1293, 378, 567],
                  ["191108", (1247,435), 1138, 1371, 305, 553],
                  ["191118", (1304,401), 1164, 1422, 291, 509],
                  ["191204", (1260,477), 1138, 1396, 378, 567],
                  ["200116", (1212,397), 1086, 1319, 291, 509],
                  ["201021_1", (1244,212), 1112, 1396, 29, 436],
                  ["201021_2", (1327,442), 1164, 1448, 276, 625],
                  ["201022_2", (1282,431), 1164, 1448, 320, 553],
                  ["201022_3", (1219,485), 1138, 1293, 407, 582],
                  ["201028_2", (1143,466), 1008, 1293, 364, 567],
                  ["201028_3", (1198,405), 1086, 1293, 262, 567],
                  ["201030", (1363,346), 1267, 1474, 218, 480],
                  ["201104", (1282,268), 1138, 1422, 87, 465],
                  ["201105", (1096,494), 983, 1241, 378, 582],
                  ["201106", (1140,264), 1008, 1267, 102, 436],
                  ["210520", (1282,421), 1138, 1422, 276, 553],
                  ["210601", (1292,468), 1138, 1422, 349, 582],
                  ["210604", (1333,454), 1241, 1422, 335, 553],
                  ["210610", (1161,451), 1008, 1267, 341, 582],
                  ["210617", (1353,418), 1267, 1435, 305, 538]]

df = pd.DataFrame(data_starfield, 
                  columns=["folder", "centre", "x_left", "x_right", "y_up", "y_down"])

screen_resolution = np.array([155, 138]) # in degrees
vf_resolution = np.array([1440, 2560])

pixels_per_degree = np.flip(vf_resolution) / screen_resolution

def gaussian(mean_pixels, sigma_pixels):
    
    Z = 0
    
    for coordinate in mean_pixels:
        
        A = 1
        x0, y0 = coordinate
        sigma_X, sigma_Y = sigma_pixels
        X, Y = np.meshgrid(np.arange(vf_resolution[1]), np.arange(vf_resolution[0]))
        
        # theta is for rotation of Gaussian blob
        theta = 0
        
        a = np.cos(theta)**2 / (2 * sigma_X ** 2) + np.sin(theta)**2 / (2 * sigma_Y ** 2)
        b = -np.sin(2 * theta) / (4 * sigma_X ** 2) + np.sin(2 * theta) / (4 * sigma_Y ** 2)
        c = np.sin(theta)**2 / (2 * sigma_X ** 2) + np.cos(theta)**2 / (2 * sigma_Y ** 2)
        
        Z += A * np.exp(-(a * (X - x0)**2 + 2*b*(X - x0)*(Y - y0) + c*(Y - y0)**2))
        
    return Z

def STMD_centres(mean, sigma, vf):
    """
    Find number of STMD centres
    """
    half_width = np.asarray([sigma, sigma]) * np.sqrt(2 * np.log(2))
    
    # Find the offset as a function of the overlap
    overlap = 0.25
    offset = 2 * (1 - overlap) * half_width
    # Create a grid
    elevation = np.arange(0, screen_resolution[1], offset[1])
    azimuth = np.arange(0, screen_resolution[0], offset[0])
    grid = np.array(list(itertools.product(elevation, azimuth)))
    
    # Re-format grid to convention
    grid[:, [0, 1]] = grid[:, [1, 0]]
    
    grid[:,0] -= (screen_resolution[0] / 2)
    grid[:,1] -= (screen_resolution[1] / 2)
    
    # Find nearest tuple to the mean
    nearest = np.array(min(grid, key=lambda x: math.hypot(x[0] - mean[0], x[1] - mean[1])))
    distance = mean - nearest
    
    points = grid + distance
    
    # Converting to pixels
    points_degrees2pixels = (points + screen_resolution / 2) * pixels_per_degree
    
    # Extracting TSDN receptive field centers
    hist = np.histogram2d(points_degrees2pixels[:,1], points_degrees2pixels[:,0], vf.shape)[0]
    TSDN_rf = vf.copy()
    TSDN_rf[TSDN_rf <= 0.5] = 0 # only take within 50% amplitude
    
    # Remove points not within TSDN receptive field
    TSDN_overlay = hist * TSDN_rf
    TSDN_coordinates = np.flip(np.stack((np.where(TSDN_overlay > 0))).T, axis=1)
    
    # fig, axes = plt.subplots(dpi=500)
    # axes.scatter(TSDN_coordinates[:,0], TSDN_coordinates[:,1])
    # axes.imshow(TSDN_rf)
    
    return TSDN_coordinates.shape[0]

def legend_for_TSDN_labels(ax, colors, labels):
    from matplotlib.lines import Line2D
    
    circle_handles = []
    for i, c in enumerate(colors):
        size = int(labels[i].split('-')[0])
        circle = Line2D([0], [0], marker='o', 
                        fillstyle='none', 
                        markeredgecolor=c, 
                        markersize=size,
                        label=labels[i],
                        linewidth=0)
        circle_handles.append(circle)
    ax.legend(handles=circle_handles)

fig, axes = plt.subplots(2, 1, figsize=(7, 9), dpi=500, sharex=True, sharey=True)
colors = ["blue", "red", "green", "orange"]
labels = ["4-7", "8-11", "12-15", "16-19"]
freq = []
for i in df.index:
    # Find average std in each axis in pixels
    x_std = np.abs(df["x_left"][i] - df["x_right"][i]) / (2 * np.sqrt(2 * np.log(2)))
    y_std = np.abs(df["y_up"][i] - df["y_down"][i]) / (2 * np.sqrt(2 * np.log(2)))
    
    # axes.errorbar(df["centre"][i][0], df["centre"][i][1], xerr=x_std, yerr=y_std)
    
    # Generate receptive field with particular TSDN mean and sigma
    vf = gaussian(mean_pixels=[df["centre"][i]], sigma_pixels=[x_std, y_std])
                  
    # Calculate number of STMD centres
    num_centres = STMD_centres([30, 55], 3, vf)
    freq.append(num_centres)
    
    # Get 50% radius of maximum
    avg_50 = np.mean([x_std, y_std]) * (2 * np.sqrt(2 * np.log(2))) / 2
    
    for a, ax in enumerate(axes):
        if a == 0:
            ax.scatter(*df["centre"][i])
            ax.set_title("Starfield TSDN RF centres - 35")
        elif a == 1:
            index = num_centres // 4 - 1
            circle = patches.Circle((df["centre"][i]), radius=avg_50, 
                                    fill=False, 
                                    edgecolor=colors[index],
                                    label=labels[index],
                                    transform=ax.transData)
            ax.add_patch(circle)
            ax.set_title("Starfield TSDN RFs - 35")
            legend_for_TSDN_labels(ax, colors, labels)
        ax.set_xlim([0, 2560])
        ax.set_ylim([1440, 0])
        
unique_freq, unique_inverse = np.unique(freq, return_inverse=True)

fig, axes = plt.subplots(dpi=500, tight_layout=True)
axes.hist(freq, bins=np.arange(unique_freq.min(), unique_freq.max()+2)-0.5)
axes.set_xticks(np.arange(unique_freq.min(), unique_freq.max()+1))
axes.set_xlabel('Number of STMD centres')
axes.set_ylabel('Frequency')