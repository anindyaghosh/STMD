import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-st', '--stim_type', type=str)

args = parser.parse_args()

stim_folder = os.path.join('starfield_stimulus', args.stim_type)
os.makedirs(stim_folder, exist_ok=True)

pre_remaining = 5
stim_frames = 160
post_remaining = 5

pre_stim = int(pre_remaining/165 * 1000)
stim_time = int(stim_frames/165 * 1000)
post_stim = int(post_remaining/165 * 1000)

total_time = pre_stim + stim_time + post_stim

def extractPoints(condition, filename):
    image = cv2.imread(os.path.join('ImageSequences', condition, filename + '.png'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # Find circles 
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    info = []
    for c in cnts:
        area = cv2.contourArea(c)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        info.append((x, y, r, round(image[round(y), round(x)][0])))
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    # cv2.imshow('image', image)
    # cv2.imwrite('captured.png', image)
    # cv2.waitKey()
        
    return info
    
start_image = extractPoints('NegativeSideslip', 'PosSideslip 11-Jun-2020 16_44_30_0081')
end_image = extractPoints('NegativeSideslip', 'PosSideslip 11-Jun-2020 16_44_30_0082')

def interpolation(start, end, upsampling_resolution):
    # Calculating velocities
    velocities = []
    for p, point in enumerate(start):
        x, y, r, intensity = point
        x_end, y_end, _, _ = end[p]
        
        velocity = x_end - x # points only move horizontally
        velocity_upsampled = np.abs(velocity * 1/upsampling_resolution * 1)
        # In case of errant velocity calculations
        if velocity_upsampled >= 10:
            velocity_upsampled = r / 10 # sizes below 10 seem to have sub-one velocity, so normalise to 10
        velocities.append(velocity_upsampled)
    return velocities

def naming_convention(i):
    # zfill pads string with zeros from leading edge until len(string) = 6
    return 'IMG' + str(i).zfill(6)

def generate_images(start, t):
    velocities = interpolation(start_image, end_image, upsampling_resolution=1000/165)
    
    # Create background
    image = np.ones((1440, 2560, 3)) * 255
    for p, point in enumerate(start):
        x, y, r, intensity = point
        
        if args.stim_type == 'alone':
            r = 0
        elif args.stim_type == 'stationary':
            velocities[p] *= 0
        elif args.stim_type == 'syn_directional':
            velocities[p] *= -1
        elif args.stim_type == 'contra_directional':
            pass
        cv2.circle(image, (round(x + velocities[p]*(t-pre_stim)), round(y)), round(r), tuple([intensity]*3), -1)
        
    # Add target
    # if t >= int(pre_stim + stim_time/2):
    #     target_timestep = t - int(pre_stim + stim_time/2)
    #     cv2.circle(image, (round(target_start[0] - target_velocity*(target_timestep+1)), 
    #                         target_start[1]), 8, tuple([0]*3), -1)
        
    cv2.imwrite(os.path.join(stim_folder, naming_convention(t+1) + '.bmp'), image)

for time in tqdm(range(total_time)):
    image = np.ones((1440, 2560, 3)) * 255
    if time < pre_stim or time > (pre_stim + stim_time):
        cv2.imwrite(os.path.join(stim_folder, naming_convention(time+1) + '.bmp'), image)
    else:
        generate_images(start_image, time)

# Write out video
# out = cv2.VideoWriter('test' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, image_array.shape[:-1])
 
# for i in range(len(image_array)):
#     out.write(image_array[i])
# out.release()