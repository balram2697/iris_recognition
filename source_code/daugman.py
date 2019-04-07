import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter

def circular_integral(image,x,y,r,start,end,n):
    s=2*pi/n
    su=0
    for i in np.arange(start,end,(end-start)/n):
        theta=i*s
        x1=int(x+r*cos(theta))
        y1=int(y+r*sin(theta))
        su=su+image[x1,y1]
    return su/n
def daugman_main(im1,im,x_pupil,y_pupil,r_pupil):      
    a=[]
    b=[]
    c=[]
    maxr=r_pupil*10
    img1 = np.pad(im, maxr, 'edge')
    for i in range(x_pupil-r_pupil,x_pupil+r_pupil,5):
        a.append(i)
    for i in range(y_pupil-r_pupil,y_pupil+r_pupil,5):
        b.append(i)
    for i in range(r_pupil+10,maxr,1):
        c.append(i)
    mx=0
    my=0
    mr=0
    mt=0
    for k in range(len(a)):
        for l in range(len(b)):
            hd=np.zeros((len(c)),np.float64)
            for g in range(len(c)):
                p=circular_integral(img1,b[l],a[k],c[g],1,3,8)+circular_integral(img1,b[l],a[k],c[g],5,7,8)
                hd[g]=p
            hd=(hd[2:]-hd[:-2])/2
            hd=gaussian_filter(hd,5)
            hd=np.abs(hd)
            maxi=np.max(hd)
            if(maxi>mt):
                mt=maxi
                mx=a[k]
                my=b[l]
                mr=c[np.argmax(hd)]
    cv2.circle(im1,(my,mx),mr,(0,0,255),2)
    cv2.imshow('daugman_method',im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x_iris_d=mx
    y_iris_d=my
    r_iris_d=mr
    return x_iris_d,y_iris_d,r_iris_d
