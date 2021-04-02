import numpy as np 
import cv2 as cv

def corr(img,mask):
    row,col =img.shape
    m,n = mask.shape
    new = np.zeros((row+m-1,col+n-1))
    n = n//2
    m=m//2
    filtered_img = np.zeros(img.shape)
    new[m:new.shape[0]-m,n:new.shape[1]-n]=img
    for i in range (m,new.shape[0]-m):
        for j in range (n,new.shape[1]-n):
            temp = new [i-m:i+m+1,j-m:j+m+1]
            result=temp*mask
            filtered_img[i-m,j-n]=result.sum()
            
    return filtered_img
def gaussian_filter (m,n,sigma):
    
    gaussian= np.zeros((m,n))
    m=m//2
    n=n//2
    
    for x in range (-m,m+1):
        for y in range (-n ,n+1):
            x1=sigma *(2*np.pi)**2
            x2=np.exp((x**2+y**2)/(2*sigma**2))
            
            gaussian[x+m,y+m]=(1/x1)*x2
    
    return gaussian
def average_filter (img):
    
# Obtain number of rows and columns
# of the image
    m, n = img.shape

# Develop Averaging filter(3, 3) mask
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9

# Convolve the 3X3 mask over the image
    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        
        
	    for j in range(1, n-1):
            
            
		    temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
		
		    img_new[i, j]= temp
		
    img_new = img_new.astype(np.uint8)
    return img_new
def median_filter (data,kernel_size):
    temp = []
    indexer= kernel_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    
    for i in range (len(data)):
        
        for j in range(len(data[0])):
            
            for z in range (kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(data)-1:
                    for c in range(kernel_size):
                        temp.append(0)
                        
                else :
                     if j + z - indexer <0 or j+indexer > len(data[0]) - 1:
                        temp.append(0)
                    
                     else :
                          for k in range (kernel_size):
                              temp.append(data[i+z - indexer ][j + k - indexer])
                        
                        
            temp.sort()
            data_final[i][j]= temp[len(temp)//2]
            temp = []

    return data_final
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