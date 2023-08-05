import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def scanRecurse(baseDir):
    for entry in os.scandir(baseDir):
        if entry.is_file():
            if entry.name.endswith('.jpg'):
                return os.path.join(baseDir, entry.name)
        else:
            return scanRecurse(entry.path)

# Read first image from all dataset folders
def image_read():
    root = os.path.join(os.getcwd(), '4496768')
    folders = [folder for folder in os.listdir(root)]
    images = []
    for f in folders:
        image_path = scanRecurse(os.path.join(root, f))
        images.append(cv2.imread(image_path))

    return [i for i in images if i is not None]

def display_image(image):
    fig, axes = plt.subplots(figsize=(48,36))
    axes.imshow(image, cmap='gray')
    
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
    plt.savefig(os.path.join(os.getcwd(), 'fourier'), bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    
def add_pseudo_random_targets(mean_magnitude):
    np.random.seed(0)
    pseudo_xy = np.random.randint([0] * 20, [[mean_magnitude.shape[0]], [mean_magnitude.shape[1]]])
    mean_magnitude[pseudo_xy[0,:], pseudo_xy[1,:]] = 0.0
    
    return mean_magnitude

# Obtain equalised images of only magnitudes
def fourier(targets=None):
    # Read images
    images = image_read()
    
    mean_magnitude = np.zeros(images[0].shape[:-1])
    for i, image in enumerate(images):
        # Extract green from BGR
        green = image[:,:,1]
        
        # Calculate zero mean-shifted FFT
        image_fourier = np.fft.fft2(green)
        image_fft = np.log(abs(np.fft.fftshift(image_fourier)))
        
        # display_image(image_fft)
        
        # Obtain only magnitude
        image_real = np.real(image_fourier)
        
        # Inverse fourier
        image_inverse_fourier = abs(np.fft.ifft2(image_real))
        
        # Histogram equalisation
        clahe = cv2.createCLAHE()
        equalised = clahe.apply(image_inverse_fourier.astype(np.uint8))
        # equalised = cv2.equalizeHist(image_inverse_fourier.astype(np.uint8))
        
        # display_image(equalised)
        # plt.close('all')
        
        # Find mean magnitude of all images
        try:
            mean_magnitude += equalised
        except:
            equalised_resize = cv2.resize(equalised, dsize=np.flip(mean_magnitude.shape), interpolation=cv2.INTER_AREA)
            mean_magnitude += equalised_resize
            
    mean_magnitude /= (i+1)
    
    if targets is not None:
        # Add pseudo random targets
        mean_magnitude = add_pseudo_random_targets(mean_magnitude)
                
    return mean_magnitude
    
# mean_magnitude = fourier()
# display_image(mean_magnitude)