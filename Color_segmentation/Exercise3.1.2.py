import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import math
import os

def main():
    path = os.path.dirname(__file__)
    img = cv2.imread(path+'/EB-02-660_0595_0414.JPG')
    assert img is not None, "Failed to load image."
    #pixels = np.reshape(img, (-1, 3))
    img_annotated = cv2.imread(path+'/EB-02-660_0595_0414_mask.JPG')
    
    mask = cv2.inRange(img_annotated, (0, 0, 200), (5, 5, 255))
    #mask_pixels = np.reshape(mask, (-1))
    cv2.imwrite(path+'/worked_img/annotated_pumkin.jpg', mask)
   
    pixel_array = np.reshape(img, (-1,3))
    mask_array = np.reshape(mask, (-1,3))

    #distance in color code
    ref_color_r = 255
    ref_color_g = 165
    ref_color_b = 0

    print(pixel_array)
    #print(mask_array)
    max_dist = 0
    min_dist = 255
    print("to loop")
    n = 0
    for x in pixel_array:
        #print(x)
        eclidian_dist=math.sqrt((ref_color_r - pixel_array[n][0])**2 + (ref_color_g - pixel_array[n][1])**2 + (ref_color_b - pixel_array[n][2])**2)
        if eclidian_dist > max_dist:
            max_dist = eclidian_dist
        
        if eclidian_dist < min_dist:
            min_dist = eclidian_dist
        n+=1
    print(min_dist)    
    print(max_dist)
    print("num of itterations")
    print(n)



main()