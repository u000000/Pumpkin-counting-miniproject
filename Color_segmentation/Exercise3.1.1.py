import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import os


def annotate(img):

    #setup blob detector
    params = cv2.SimpleBlobDetector_Params()
 
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector(params)
    print("test2")

    keypoints = detector.detect(img)
    print("test2.5")
    img_annotated = cv2.drawKeypoints(img,keypoints, np.array([]), (0, 160, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("test3")
    # Show annotated img
    cv2.imshow("Keypoints", img_annotated)
    return img_annotated


def main():
    path = os.path.dirname(__file__)
    print(path+'/EB-02-660_0595_0414.JPG')
    img = cv2.imread(path+'/EB-02-660_0595_0414.JPG')
    assert img is not None, "Failed to load image."
    #pixels = np.reshape(img, (-1, 3))
    img_annotated = cv2.imread(path+'/EB-02-660_0595_0414_mask.JPG')
    
    mask = cv2.inRange(img_annotated, (0, 0, 200), (5, 5, 255))
    #mask_pixels = np.reshape(mask, (-1))
    cv2.imwrite(path+'/worked_img/annotated_pumkin.jpg', mask)

    # Determine mean value, standard deviations and covariance matrix
    # for the annotated pixels.
    # Using cv2 to calculate mean and standard deviations
    mean_bgr, std_bgr = cv2.meanStdDev(img, mask = mask)
    print("rgb")
    print("Mean color values of the annotated pixels")
    print(mean_bgr)
    print("Standard deviation of color values of the annotated pixels")
    print(std_bgr)

    color_mask = cv2.bitwise_and(img,img,mask=mask)

    cv2.imwrite(path+'/worked_img/annotated_pumkin_color.jpg', color_mask)

    b, g, r = cv2.split(color_mask)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = cv2.cvtColor(color_mask,cv2.COLOR_BGR2RGB).reshape((np.shape(color_mask))[0]*np.shape(color_mask)[1], 3)
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    # axis.set_zlabel("Blue")
    # plt.show()
    plt.savefig(path+'/worked_img/StandardDeviationOfColorValues.png')
    print('BRG done!')



    # # CIE L*a*b
    # labimg = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

    # print("lab")
    # mean_lab, std_lab = cv2.meanStdDev(labimg, mask = mask)
    # print("Mean color values of the annotated pixels")
    # print(mean_lab)
    # print("Standard deviation of color values of the annotated pixels")
    # print(std_lab)

    # color_mask_lab = cv2.bitwise_and(labimg,labimg,mask=mask)

    # cv2.imwrite(path+'/worked_img/annotated_pumkin_color_lab.jpg', color_mask_lab)
    
    





main()