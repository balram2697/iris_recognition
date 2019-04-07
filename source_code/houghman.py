import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter

def houghman_main(im1,x_pupil,y_pupil,r_pupil,edged):
    M,N=edged.shape
    a=[]
    b=[]
    c=[]
    maxr=r_pupil*10;
    for i in range(x_pupil-r_pupil,x_pupil+r_pupil,5):
        a.append(i)
    for i in range(y_pupil-r_pupil,y_pupil+r_pupil,5):
        b.append(i)
    for i in range(r_pupil+10,maxr,1):
        c.append(i)
    accumalator=np.zeros((len(a),len(b),len(c)))
    mx=0;
    x_iris=0;
    y_iris=0;
    r_iris=0;
    for i in range(M):
        for j in range(N):
            if edged[i][j]>0:
                for k in range(len(a)):
                    for l in range(len(b)):
                        R=round(math.sqrt((i-a[k])*(i-a[k])+(j-b[l])*(j-b[l])))
                        o=-1
                        for g in range(len(c)):
                            if R==c[g]:
                                o=g;
                        if o!=-1:
                            accumalator[k][l][o]=accumalator[k][l][o]+1
                            if accumalator[k][l][o]>mx:
                                x_iris=k
                                y_iris=l
                                r_iris=R
                                mx=accumalator[k][l][o]

    cv2.circle(im1,(b[y_iris],a[x_iris]),r_iris,(0,255,0),2)
    cv2.imshow('houghman_method',im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    y_iris=b[y_iris]
    x_iris=a[x_iris]
    return x_iris,y_iris,r_iris