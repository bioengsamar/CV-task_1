from PyQt5 import QtWidgets
from design4 import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot

import sys
import pyqtgraph as pg
import numpy as np
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
from PyQt5.QtGui import QPixmap
import functions
#from imageModel import ImageModel
#from modesEnum import Modes

 

# replace <...> with your py file generate from pyuic5 command - without .py-

# It's preferaple to work on MainWindow in qt designer to support layouts
# use Dialog for relatively small windows or windows that don't have too much elements  


class ApplicationWindow(QtWidgets.QMainWindow):
    addimage=0
    
   
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
    def openfile(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file',
                                            '/home/youssef/', "Image files (*.jpg *.gif *.png)")
        self.imagePath= filename[0]
        
        self.openimage(self.imagePath)
        
    def openimage(self,imagePath):        
        self.addimage=1
        self.imgByte = cv.imread(self.imagePath)

        pixmap = QPixmap(imagePath).scaled(400,120)
        
        self.ui.image.setPixmap(QPixmap(pixmap))
 
    def resultingimage(self,mode):
        if self.addimage==1:            
            if mode==0:
                if self.ui.comboBox.currentText()=='combo1':
                    print('c1')
                if self.ui.comboBox.currentText()=='combo2':
                    print('c2')    
            #print(self.ui.comboBox.currentText())

            elif mode==1:
                if self.ui.filter.currentText()=='average filter':
                    self.gray= cv.cvtColor(self.imgByte, cv.COLOR_BGR2GRAY)
                    self.avg=functions.average_filter(self.gray)
                    result_image=cv.imwrite("result.jpg",self.avg)
                    result = QPixmap("result.jpg").scaled(400,120)
                    self.ui.image_result.setPixmap(QPixmap(result))
                
                
                
                elif self.ui.filter.currentText()=='gaussian filter':
                    g=functions.gaussian_filter(5,5,2)
                    self.gray= cv.cvtColor(self.imgByte, cv.COLOR_BGR2GRAY)
                    self.n=functions.corr(self.gray,g)
                    result_image=cv.imwrite("result.jpg",self.n)
                    result = QPixmap("result.jpg").scaled(400,120)
                    self.ui.image_result.setPixmap(QPixmap(result))
                elif self.ui.filter.currentText()=='median filter':
                    self.gray= cv.cvtColor(self.imgByte, cv.COLOR_BGR2GRAY)
                    self.med=functions.median_filter(self.gray,3)
                    result_image=cv.imwrite("result.jpg",self.med)
                    result = QPixmap("result.jpg").scaled(400,120)
                    self.ui.image_result.setPixmap(QPixmap(result))
                    
                
                
        
        
            elif mode==5:
            
# =============================================================================
#             self.gray_image = (self.imgByte[:,:,0] + self.imgByte[:,:,1] + self.imgByte[:,:,2]) / 3
#             height, width, channels = self.imgByte.shape
#  
#             for i in range(height):
#                 
#                 for j in range(width):
#                     self.gray_image[i,j] = 0.3 * self.imgByte[i,j][0] + 0.59 * self.imgByte[i,j][1] +  0.11 * self.imgByte[i,j][2]
# =============================================================================
 
                self.gray_image=np.dot(self.imgByte[...,:3], [0.2989, 0.5870, 0.1140])

                result_image=cv.imwrite("result.jpg",self.gray_image)
                result = QPixmap("result.jpg").scaled(400,120)
                self.ui.image_result.setPixmap(QPixmap(result))
            elif mode==6:                 
                if self.ui.freq_filter.currentText()=='HPF':
                    self.high=functions.low_high_pass(self.imgByte,high_pass= True)
                    result_image=cv.imwrite("result.jpg",self.high.astype('uint8'))
                    result = QPixmap("result.jpg").scaled(400,120)
                    self.ui.image_result.setPixmap(QPixmap(result))
                      
                elif self.ui.freq_filter.currentText()=='LPF':
                    self.low=functions.low_high_pass(self.imgByte)
                    result_image=cv.imwrite("result.jpg",self.low)
                    result = QPixmap("result.jpg").scaled(400,120)
                    self.ui.image_result.setPixmap(QPixmap(result))
          
                    
                


            else:
                main()
        else:            
            print('no change')
     
    def histogram_RGB(self):
        if self.addimage==1:
            if self.ui.histogram.currentText()=='R,G,B histogram':
                colors = ("r", "g", "b")
                channel_ids = (0, 1, 2)
                print((colors[0]))
                #plt.xlim([0, 256])
                for channel_id, c in zip(channel_ids, colors):
                    histogram, bin_edges = np.histogram(
                            self.imgByte[:, :, channel_id], bins=256, range=(0, 256))
                    #self.rgb=np.dstack(())

                    #plt.plot(bin_edges[0:-1], histogram, color=c)
                    self.ui.graph.plot(bin_edges[0:-1], histogram,pen=c)
                    

        



            
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.ui.upload.clicked.connect(application.openfile)
    #application.ui.noise.activated[str].connect(lambda:application.resultingimage(0)) 
    application.ui.filter.activated[str].connect(lambda:application.resultingimage(1)) 
    #application.ui.edge.activated[str].connect(lambda:application.resultingimage(2)) 
    #application.ui.equalize.clicked.connect(lambda:application.resultingimage(3))
    #application.ui.normalize.clicked.connect(lambda:application.resultingimage(4))
    application.ui.gray_scale.clicked.connect(lambda:application.resultingimage(5))    
    application.ui.freq_filter.activated[str].connect(lambda:application.resultingimage(6)) 
    #application.ui.hybrid.clicked.connect(lambda:application.resultingimage(7))
    application.ui.histogram.activated[str].connect(application.histogram_RGB) 




   
#/usr/bin/python2.7

    application.show()
    app.exec_()


if __name__ == "__main__":
    main()