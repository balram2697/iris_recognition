import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter
from pupil import pupil_main
from canny import canny_main
from houghman import houghman_main
from daugman import daugman_main
from normalise import normalise_main


im1=cv2.imread('C:/Users/Balram Choudhary/Desktop/eye1.jpg',1)
im=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(im,5)
x_pupil,y_pupil,r_pupil=pupil_main(image,im1)
edged=canny_main(image)
x_iris,y_iris,r_iris=houghman_main(im1,x_pupil,y_pupil,r_pupil,edged)
x_iris_d,y_iris_d,r_iris_d=daugman_main(im1,im,x_pupil,y_pupil,r_pupil)
normalized_image=normalise_main(im,x_pupil,y_pupil,r_pupil,x_iris,y_iris,r_iris)
