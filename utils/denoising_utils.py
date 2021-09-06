import os
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    num_stripe = 5 # number of stripes in each bandwidth

    # add Gaussian noise
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    # add stripe noise
    for i in range(img_noisy_np.shape[0]):
        rand_stripe = np.random.randint(img_noisy_np.shape[1], size=num_stripe)
        for j in range(num_stripe):
            img_noisy_np[i, rand_stripe[j], :] = 0

    return img_noisy_np