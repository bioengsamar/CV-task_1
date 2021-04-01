import numpy as np
#from scipy import ndimage
import cv2
import argparse
import matplotlib.pyplot as plt
from convolution import convolution

roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = image.astype('float64')
#image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
#image /= 255

vertical = convolution( image, roberts_cross_v )
horizontal = convolution( image, roberts_cross_h )

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img *= 255.0 / edged_img.max()
#edged_img /= 255

plt.imshow(edged_img, cmap='gray')
plt.show()