import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from convolution import convolution


def sobel_edge_detection(image, mask, verbose=False):
    new_image_x = convolution(image, mask)
    new_image_y = convolution(image, np.flip(mask.T, axis=0))
    print(type(new_image_x))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    print(gradient_magnitude.shape)
 
    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()
 
    return gradient_magnitude
 
 
if __name__ == '__main__':
    mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
 
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    
    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(mask.shape)
    
    
    sobel_edge_detection(image, mask, verbose=True)