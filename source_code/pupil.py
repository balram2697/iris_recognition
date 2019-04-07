import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter

def imcomplement(image):
    minval = np.iinfo(image.dtype).min
    maxval = np.iinfo(image.dtype).max
    return minval + maxval - image
def fill(image,h_max=255):
    a = np.copy(image) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    mask = sp.ndimage.binary_erosion(~np.isnan(a), structure=el)
    out = np.copy(a)
    out[mask]=h_max
    u = np.copy(a)
    u.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(u, out):
        u = np.copy(out)
        out = np.maximum(a,sp.ndimage.grey_erosion(out, size=(3,3), footprint=el))
    return out
def pupil_main(image,im1):
    image=imcomplement(image)
    image=fill(image)
    image=imcomplement(image)
    cv2.imshow('morph_operation',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist,bins = np.histogram(image.ravel(),5,[0,256])
    threshold=bins[1]
    ret,th1 = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)
    img = cv2.bitwise_not(th1)
    kernel = np.ones((5,5),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    cv2.imshow('thresholded_image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, labels = cv2.connectedComponents(img)
    maxarea=0
    for region in regionprops(labels):
          if region.area>maxarea and region.eccentricity<=0.7:
              # print(region.centroid)
              # print(region.area)
              maxarea=region.area
              centroid = region.centroid
              box=region.bbox
    x_pupil=int(centroid[0])
    y_pupil=int(centroid[1])
    r_pupil=int(max((box[3]-box[1])/2,(box[2]-box[0])/2))
    cv2.circle(im1,(y_pupil,x_pupil),r_pupil,(255,0,0),2)
    cv2.imshow('pupil',im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x_pupil,y_pupil,r_pupil