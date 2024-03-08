import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

"""# import photo """
path = os.path.dirname(__file__)
photoName = sys.argv[1]
img_BGR = cv2.imread(path+'/'+photoName)
# cv2.imshow('img', img_BGR)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Convert image to CieLab
img = cv2.cvtColor(img_BGR,cv2.COLOR_BGR2Lab)
# cv2.imshow('img_lab', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# """# resize """
# h, w, c = img.shape
# down_points = (int(round(w/1, 0)), int(round(h/1, 0)))
# img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
# cv2.imshow('img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""# filtering """
# img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
# cv2.imshow('img_gauss', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# I don't like how it works, I loose several pumpkins under the leaves

# img = cv2.medianBlur(img, 5)
# cv2.imshow('img_median', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# I don't like how it works, I loose several pumpkins under the leaves

"""# use inRange """
# Mean color values od pumpkins: [225.69843634, 128.99106478, 176.34921817]
mean_color = np.array([225.69843634, 128.99106478, 176.34921817])

# Standard deviation of color values of the annotated pixels: [17.55660703, 10.69080414, 9.59233129]
# standard_deviation = np.array([17.55660703, 10.69080414, 9.59233129])
standard_deviation = np.array([17.55660703, 10.69080414, 9.59233129])

lower_pumpkin = np.maximum(mean_color - 2 * standard_deviation, 0)
upper_pumpkin = np.maximum(mean_color + 2 * standard_deviation, 255)

# Masking
mask = cv2.inRange(img, lower_pumpkin, upper_pumpkin)

# Mask on original img
img_masked = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('img_masked', img_masked)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""# filtering """
img_dil = cv2.dilate(mask, np.ones((1, 1), np.uint8))
# cv2.imshow('img_dil', img_dil)
# cv2.waitKey()
# cv2.destroyAllWindows()
img_ero = cv2.erode(img_dil, np.ones((20, 20), np.uint8))
# cv2.imshow('img_ero', img_ero)
# cv2.waitKey()
# cv2.destroyAllWindows()

"""# counting pumpkins """
conts, hierarchy = cv2.findContours(img_ero, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_BGR, conts, -1, (255, 0, 0), 1)
cv2.imshow('img_BGR', img_BGR)
cv2.waitKey()
cv2.destroyAllWindows()

number_of_pumpkins = 0
height, width = img.shape[:2]
centers = []
for cont in conts:
    M = cv2.moments(cont)
    x0 = int(M['m10']/M['m00'])
    y0 = int(M['m01']/M['m00'])
    centers.append((x0, y0))
    partlyCutPumpkin = False

    # for point in cont:
    #     x, y = point[0]
    #     print('x: ' + str(x))
    #     print('y: ' + str(y))
    #     if x == 0 or y == 0 or x == width or y == height:
    #         partlyCutPumpkin = True
    #     else:
    #         continue
    # if not partlyCutPumpkin:
    #     number_of_pumpkins += 1
    #     print('add')

for center in centers:
    cv2.circle(img_BGR, center, 3, (0, 0, 255), 2)

cv2.imshow('img_BGR', img_BGR)
cv2.waitKey()
cv2.destroyAllWindows()

print('Number of pumpkins: ' + str(number_of_pumpkins))