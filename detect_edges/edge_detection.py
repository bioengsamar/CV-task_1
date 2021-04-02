import numpy as np
import cv2

def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    print("Output Image size : {}".format(output.shape))
    return output

def edge_detection(image, mask_cross_v, mask_cross_h):
    new_image_x = convolution(image, mask_cross_v)
    new_image_y = convolution(image, mask_cross_h)
    print(type(new_image_x))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude

if __name__ == '__main__':
     image = cv2.imread('input.png',0)
     mask_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
     sobel_edge_detection= edge_detection(image, mask_sobel, np.flip(mask_sobel.T, axis=0) )
     
     roberts_cross_v = np.array( [[ 0, 0, 0 ], [ 0, 1, 0 ],[ 0, 0,-1 ]] )
     roberts_cross_h = np.array( [[ 0, 0, 0 ],[ 0, 0, 1 ],[ 0,-1, 0 ]] )
     roberts_edge_detection= edge_detection(image, roberts_cross_v, roberts_cross_h )
     
     prewitt_cross_v = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
     prewitt_cross_h = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
     prewitt_edge_detection= edge_detection(image, prewitt_cross_v, prewitt_cross_h )
     
     cv2.imshow('sobel edge detection', sobel_edge_detection.astype('uint8'))
     cv2.imshow('roberts edge detection', roberts_edge_detection.astype('uint8'))
     cv2.imshow('prewitt edge detection', prewitt_edge_detection.astype('uint8'))
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     
     
    