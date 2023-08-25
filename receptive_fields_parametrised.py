"""Receptive fields
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json

from receptive_field_array import RF_array

neurons = {"TSDN" : {"sigma_vals":12.74, "centre":[0, 45]}, 
           "dSTMD" : {"sigma_vals":3, "centre":[0, 55]}, 
           "wSTMD" : {"sigma_vals":[16.99, 16.99], "centre":[5, 55]}}
           # "wSTMD" : {"sigma_vals":[21.23, 16.99], "centre":[0, 55]}}

class receptive_fields:
    def __init__(self, vf_resolution):
        self.vf_resolution = vf_resolution
        self.screen_resolution = np.array([(155 * 2), 138]) # in degrees
        
        self.pixels_per_degree = np.flip(self.vf_resolution) / self.screen_resolution
        
    def check_integrity(self):
        self.reset_flag = False
        if not os.path.isfile('rf_metadata.json'):
            json.dump([self.vf_resolution.tolist(), self.screen_resolution.tolist(), neurons], open('rf_metadata.json', "w"))
        else:
            md = json.load(open('rf_metadata.json',))
            vf_resolution, screen_resolution, neurons_check = md
            
            # Compare json dict values to current input
            self.reset_flag = not (set(vf_resolution) == set(self.vf_resolution) and
                                   set(screen_resolution) == set(self.screen_resolution) and
                                   neurons_check == neurons)
            
            if self.reset_flag:
                # Save new parameters to json file
                json.dump([self.vf_resolution, self.screen_resolution.tolist(), neurons], open('rf_metadata.json', "w"))
                
    def calc_pixel_size(self, value: list):
        # Sigma is in degrees
        return value * self.pixels_per_degree # sigma_pixel = (sigma in degrees (0.59))*pixels_per_degree

    def calc_centre_coordinates(self, location : list or np.ndarray, _RF_flag : bool = False):
        # Location is in degrees (x, y) -> (column, row) where x and y are elements of (-180, 180)
        location_in_pixels = location * self.pixels_per_degree
        vf_centre = np.flip(self.vf_resolution)/2
        
        if not _RF_flag:
            x = int(vf_centre[0] + location_in_pixels[0])
            y = int(vf_centre[1] - location_in_pixels[1])
            return (x, y)
        else:
            x = (vf_centre[0] + location_in_pixels[:,0]).astype(int)
            y = (vf_centre[1] - location_in_pixels[:,1]).astype(int)
            return np.stack((x, y)).T
    
    def rf_imshow(self, rf):
        fig, axes = plt.subplots(figsize=(12,9), dpi=200)
        x_extent, y_extent = self.screen_resolution / 2
        img = axes.imshow(rf, extent=[-x_extent, x_extent, -y_extent, y_extent])
        axes.grid()
        axes.set_xlabel('Azimuth [$^\circ$]')
        axes.set_ylabel('Elevation [$^\circ$]')
        plt.colorbar(img)

    def gaussian(self, neuron : str, vf : np.ndarray):
        """For the general form of the Gaussian function, the coefficient A is the height of the peak and (x0, y0) is the center of the Gaussian blob.
    
        Parameters
        ----------
        neuron: str
            contains attributes of mean and sigma in x and y directions (in degrees).
    
        Returns
        -------
        Receptive field.
    
        """
        
        mean, sigma = neurons[neuron]["centre"], neurons[neuron]["sigma_vals"]
        
        if isinstance(sigma, (float, int)):
            sigma = [sigma, sigma]
        
        if neuron == "dSTMD":
            coords = RF_array(mean=mean, sigma=sigma, overlap=0.25, screen_resolution=self.screen_resolution, vf=vf)
            # Remove all points below zero degrees elevation
            coords = coords[coords[:,1] >= 0]
            _RF_flag = True
            mean_pixels = self.calc_centre_coordinates(coords, _RF_flag)
        else:
            mean_pixels = [self.calc_centre_coordinates(mean)]
        
        sigma_pixels = self.calc_pixel_size(sigma)
        
        Z = 0
        
        for coordinate in mean_pixels:
            
            A = 1
            x0, y0 = coordinate
            sigma_X, sigma_Y = sigma_pixels
            X, Y = np.meshgrid(np.arange(self.vf_resolution[1]), np.arange(self.vf_resolution[0]))
            
            # theta is for rotation of Gaussian blob
            theta = 0
            
            a = np.cos(theta)**2 / (2 * sigma_X ** 2) + np.sin(theta)**2 / (2 * sigma_Y ** 2)
            b = -np.sin(2 * theta) / (4 * sigma_X ** 2) + np.sin(2 * theta) / (4 * sigma_Y ** 2)
            c = np.sin(theta)**2 / (2 * sigma_X ** 2) + np.cos(theta)**2 / (2 * sigma_Y ** 2)
            
            Z += A * np.exp(-(a * (X - x0)**2 + 2*b*(X - x0)*(Y - y0) + c*(Y - y0)**2))
            
            # if neuron == "wSTMD":
            #     Z[:,:x0] = 0
            
        return Z

    def run(self):
        self.check_integrity()
        if not os.path.isfile('neurons_rf.npz') or self.reset_flag:
            vf = np.zeros((*self.vf_resolution, 3))
            for i, neuron in enumerate(neurons.keys()):
                vf[...,i] += self.gaussian(neuron, vf)
                # Clip values to ensure multiplication of response * RF = 0 when stimulus is not close to the centre of the RF.
                vf[...,i][vf[...,i] <= 0.01] = 0
                self.rf_imshow(vf[...,i])
            # Take half of receptive field to simulate right eye
            vf = vf[:,int(vf.shape[1] / 2):,:]
            np.savez('neurons_rf.npz', vf=vf)
        else:
            vf = np.load('neurons_rf.npz')["vf"]
        return vf
    
# vf = receptive_fields((1440, 2560)).run()