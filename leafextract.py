# -*- coding: utf-8 -*-
"""
Authors : Kithmin Wickramaisnghe and Naveen Karunanayake

///////    Leaf Extraction from images using openCV    ////////
"""

import cv2 as cv
import numpy as np
from common import Sketcher
#from matplotlib import pyplot as plt

im0 = cv.imread('coins3.jpg')
im0 = cv.imread('Test1.jpg')
#im0 = cv.imread('Test2.jpg')

gray1 = cv.cvtColor(im0,cv.COLOR_BGR2GRAY)
cv.imshow('image1',gray1)

####### LEAF MARKER ############

#Using Otsu's thresholding after gaussian filtering 
blur = cv.GaussianBlur(gray1,(5,5),0)
ret, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#noise removal using grayscale morphology
kernel = np.ones((3,3),np.uint8)

closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations = 5)
cv.imshow('image',closing)
opening = cv.morphologyEx(closing,cv.MORPH_OPEN,kernel,iterations = 3)
cv.imshow('image2',opening)

#sure background area
sure_bg = cv.dilate(opening,kernel,iterations = 2)
cv.imshow('image5',sure_bg)
#Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
cv.imshow('image3',dist_transform)

#Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow('image4',unknown)

#Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
cv.imshow('image6',sure_fg)
# Add one to all labels so that sure bacground is not 0, but 1
markers = markers+1


# Now, mark the region of unknown with zero
markers[unknown==255] = 0
cv.imshow('image7',unknown)

markers = cv.watershed(im0,markers)
im0[markers == -1] = [255,0,0]

cv.imshow('image8',im0)
gray2 = cv.cvtColor(im0,cv.COLOR_BGR2GRAY)
im_color = cv.applyColorMap(gray2, cv.COLORMAP_JET)

cv.imshow('image9',im_color)
k = cv.waitKey(0) & 0xFF
if k == 27: #wait for ESC to exit
    cv.destroyAllWindows()
elif k == ord('s'): #wait for 's' key to save and exit
    cv.imwrite('result.jpg',im0)
    cv.destroyAllWindows()
    

#w = watershed();
