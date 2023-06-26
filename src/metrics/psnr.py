import math
import cv2 
import numpy as np 
  

def compute(original:np.ndarray, compressed:np.ndarray) -> float: 
    """Compute the PSNR metric of a sample by providing a reference and the sample

    Args:
        original (np.ndarray): [description]
        compressed (np.ndarray): [description]

    Returns:
        float: [description]
    """

    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse)) 
    return psnr 
