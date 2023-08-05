import cv2
import numpy as np

def sparse_dot_field():
    image = np.ones([8600, 8600]) * 255
    
    # Stimulus generation
    chars = np.random.randint([0, 0, 0, 1], [*image.shape, 255, 100], size=(1000, 4))
    
    for i in chars:
        image[i[0]:i[0]+i[3], i[1]:i[1]+i[3]] = i[2]
        
    cv2.imwrite('sparse_field.png', image)
    
def clouds():
    image = np.ones([8600, 8600]) * 255
    
    #TODO: create patches of circles