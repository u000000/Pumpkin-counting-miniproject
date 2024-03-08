import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import os



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

    # b, g, r = cv2.split(color_mask)
    flat_color_mask = color_mask.reshape((np.shape(color_mask))[0]*np.shape(color_mask)[1], 3)
    flat_mask = mask.reshape((np.shape(mask))[0]*np.shape(mask)[1])
    bool_mask = np.array(flat_mask, dtype=bool)
    bool_mask_inv = np.invert(bool_mask)

    only_color = np.delete(flat_color_mask,bool_mask_inv,0)

    only_color_rot = np.rot90(only_color)
    b, g, r = [only_color_rot[0],only_color_rot[1],only_color_rot[2]]
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    only_color = only_color.reshape(np.shape(only_color)[0],1,3)
    print(np.shape(only_color))

    pixel_colors = cv2.cvtColor(only_color,cv2.COLOR_BGR2RGB)
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    pixel_colors = np.squeeze(pixel_colors)


    print(np.shape(b))
    print(np.shape(g))
    print(np.shape(r))
    print(np.shape(pixel_colors))
    # exit()
    
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    # plt.show()
    plt.savefig(path+'/worked_img/StandardDeviationOfColorValues.png')
    print('BRG done!')



    # # CIE L*a*b


    labimg = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

    print("lab")
    mean, std = cv2.meanStdDev(labimg, mask = mask)
    print("Mean color values of the annotated pixels")
    print(mean)
    print("Standard deviation of color values of the annotated pixels")
    print(std)

    
    # color_mask = cv2.bitwise_and(img,img,mask=mask)
    # cv2.imwrite(path+'worked_img/annotated_pumkin_color.jpg', color_mask)

    print(only_color.shape)

    only_color_lab = cv2.cvtColor(only_color,cv2.COLOR_BGR2Lab)
    
    print(only_color_lab.shape)

    only_color_lab = np.squeeze(only_color_lab)

    print(only_color_lab.shape)

    only_color_lab_rot = np.rot90(only_color_lab)
    l, a, b = [only_color_lab_rot[0],only_color_lab_rot[1],only_color_lab_rot[2]]

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(l.flatten(), a.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("l")
    axis.set_ylabel("a")
    axis.set_zlabel("b")
    plt.show()


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