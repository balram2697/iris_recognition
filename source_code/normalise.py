import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter


def correct(im,x,y):
    a,b=im.shape
    if 0<=x<=b and 0<=y<=a:
        return True
    else:
        return False
def normalise_main(im,x_pupil,y_pupil,r_pupil,x_iris,y_iris,r_iris):   
    radial_resolution=64 #no of data points
    angular_resolution=256 #no of radial lines
    angles=angular_resolution
    radius=radial_resolution+2
    #calculated around pupil centre
    if x_pupil>x_iris:
        ox=x_pupil-x_iris
    else:
        ox=x_iris-x_pupil
    if y_pupil>y_iris:
        oy=y_pupil-y_iris
    else:
        oy=y_iris-y_pupil  
    if ox!=0:
        u=atan(oy/ox)
    else:
        u=pi/2.0
    alpha=ox*ox+oy*oy
    mask_image=np.empty((radius-2,angles),np.uint8) 
    normalise_image=np.empty((radius-2,angles),np.uint8)
    for i in range(2):
        theta=(i*(2*pi))/(angles)
        costheta=cos(theta)
        sintheta=sin(theta)
        beta=cos(pi-u-theta)
        R=sqrt(alpha)*beta+sqrt(alpha*beta*beta-(alpha-r_iris*r_iris))
        R=R-r_pupil
        for j in range(radius):
            r=r_pupil+(R-j)/(radius-1)
            x=int(r_pupil+r*costheta)
            y=int(r_pupil+r*sintheta)
            val=im[y][x]
            if 0<j<radius-1:
                normalise_image[j-1][i]=val
                if correct(im,x,y)==False:
                    mask_image[j-1][i]=0
                elif val<=40:
                    mask_image[j-1][i]=0
                elif val>=220:
                    mask_image[j-1][i]=0
                else:
                    mask_image[j-1][i]=1
    normalized=np.multiply(normalise_image,mask_image)
    cv2.imshow('normalise_image',normalise_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return normalized
