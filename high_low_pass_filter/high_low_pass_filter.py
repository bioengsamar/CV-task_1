import numpy as np
import cv2

def draw_cicle(shape,diamiter):

    assert len(shape) == 2
    TF = np.zeros(shape,dtype=np.bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

def filter_circle(TFcircleIN,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)

def low_high_pass(img, high_pass= False):
    #low pass filter
    TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
    #high pass filter
    TFcircleOUT  = ~TFcircleIN
    
    fft_img_filtered_IN = []
    fft_img_filtered_OUT = []
    fft_img = np.zeros_like(img,dtype=complex)
    for ichannel in range(fft_img.shape[2]):
        fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(img[:,:,ichannel]))
    
     ## for each channel, pass filter
    for ichannel in range(fft_img.shape[2]):
        fft_img_channel  = fft_img[:,:,ichannel]
       
        ## circle IN >> low pass filter
        temp = filter_circle(TFcircleIN,fft_img_channel)
        fft_img_filtered_IN.append(temp)
        if high_pass:
            ## circle OUT>> high pass filter
            temp = filter_circle(TFcircleOUT,fft_img_channel)
            fft_img_filtered_OUT.append(temp) 
        
    
    if high_pass:
        fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
        fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))
        img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)
        return img_reco_filtered_OUT
    
    fft_img_filtered_IN = np.array(fft_img_filtered_IN)
    fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
    img_reco_filtered_IN  = inv_FFT_all_channel(fft_img_filtered_IN)
    return img_reco_filtered_IN

if __name__ == '__main__':
    
    img = cv2.imread('input.png')
    
    
    cv2.imshow('original image', img)
    cv2.imshow('low pass filter image', np.abs(low_high_pass(img)).astype('uint8'))
    cv2.imshow('high pass filtered image', np.abs(low_high_pass(img, high_pass=True)).astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    