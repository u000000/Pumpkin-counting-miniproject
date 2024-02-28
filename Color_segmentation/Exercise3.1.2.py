import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
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

    # Determine mean value, standard deviations and covariance matrix
    # for the annotated pixels.
    # Using cv2 to calculate mean and standard deviations
    mean, std = cv2.meanStdDev(img, mask = mask)
    print("rgb")
    print("Mean color values of the annotated pixels")
    print(mean)
    print("Standard deviation of color values of the annotated pixels")
    print(std)

    labimg = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

    print("lab")
    mean, std = cv2.meanStdDev(labimg, mask = mask)
    print("Mean color values of the annotated pixels")
    print(mean)
    print("Standard deviation of color values of the annotated pixels")
    print(std)

    color_mask = cv2.bitwise_and(img,img,mask=mask)

    cv2.imwrite('worked_img/annotated_pumkin_color.jpg', color_mask)


    r, g, b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    # axis.set_zlabel("Blue")
    #plt.show()




main()