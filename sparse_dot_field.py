import cv2
import numpy as np

def sparse_dot_field(uniform=False):
    image = np.ones([8600, 8600]) * 255
    
    # Stimulus generation
    chars = np.random.randint([0, 0, 0, 1], [*image.shape, 255, 100], size=(5000, 4))
    
    for i in chars:
        if uniform:
            size = 15
            filename = 'sparse_field_uniform.png'
        else:
            size = i[3]
            filename = 'sparse_field.png'
        image[i[0]:i[0]+size, i[1]:i[1]+size] = i[2]
        
    cv2.imwrite(filename, image)
    
def clouds():
    image = np.ones([8600, 8600]) * 255
    
    #TODO: create patches of circles
    
sparse_dot_field(uniform=True)