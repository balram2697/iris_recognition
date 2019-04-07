import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
import scipy as sp
import scipy.ndimage
from math import pi, sin, cos, atan, sqrt
from scipy.ndimage.filters import gaussian_filter

def canny_main(image):
	smooth =image
	for i in range(200):
	    smooth=cv2.medianBlur(smooth,5)
	v = np.median(smooth)
	lower = int(max(0, (1.0 - 0.33) * v))
	upper = int(min(255, (1.0 + 0.33) * v))
	edged = cv2.Canny(smooth, lower, upper)
	cv2.imshow('canny',edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return edged