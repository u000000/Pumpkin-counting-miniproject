import cv2 as cv
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import Window,transform
from rasterio import DatasetReader
import os
from typing import Tuple
from math import sqrt



class tile_maneger:

    def __init__(self,file_path, tile_width, tile_height, overlap) -> None:
        self.file :DatasetReader = rasterio.open(file_path)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.ncols, self.nrows = self.file.width, self.file.height
        self.xstep = self.tile_width - overlap
        self.ystep = self.tile_height - overlap
        self.x = 0
        self.y = 0
        print(f"width:{self.ncols}  height:{self.nrows}")

    def get_total_tile_count(self) -> int:
        tiles_in_x = int(self.ncols / self.xstep)
        if self.ncols % self.xstep != 0:
            tiles_in_x = tiles_in_x + 1
        tiles_in_y = int(self.nrows / self.ystep)
        if self.nrows % self.ystep != 0:
            tiles_in_y = tiles_in_y + 1    
        return tiles_in_x*tiles_in_y
    
    def get_grid_shape(self) -> int:
        tiles_in_x = int(self.ncols / self.xstep)
        if self.ncols % self.xstep != 0:
            tiles_in_x = tiles_in_x + 1
        tiles_in_y = int(self.nrows / self.ystep)
        if self.nrows % self.ystep != 0:
            tiles_in_y = tiles_in_y + 1    
        return (tiles_in_x,tiles_in_y)

    def get_next_tile(self) -> Tuple[any,bool]:
        is_last = False
        if (self.y > self.nrows):
                raise "no more tiles"

        window = Window.from_slices(slice(self.y, self.y+self.tile_height),slice(self.x,self.x+self.tile_width))
        # window_transform = transform(window, self.file.transform)
        # print(window)
        data = self.file.read(window=window,boundless=True)


        if (self.x + self.xstep < self.ncols):
            self.x = self.x + self.xstep
        else:
            self.x = 0
            self.y = self.y + self.ystep


        # print(f"width:{self.ncols}  height:{self.nrows}")
        # print(f"x:{self.x}  y:{self.y}")

        if (self.y > self.nrows):
            is_last = True


        return data,is_last
    
def inEclidianDist(img,mean,max_dist):
    
    [l,a,b] = cv.split(img)
    l = l - mean[0]
    a = a - mean[1]
    b = b - mean[2] 

    eclidian_dist = np.sqrt(l*l + a*a + b*b)
    
    return cv.inRange(eclidian_dist,0,max_dist)


def count_pumkins(img) -> int:

    img = cv.cvtColor(img,cv.COLOR_BGR2Lab)
    # Masking
    # mask = cv.inRange(img, lower_pumpkin, upper_pumpkin)
    mask = inEclidianDist(img, [225.69843634,128.99106478,176.34921817,], 20)

    # Mask on original img
    # img_masked = cv.bitwise_and(img, img, mask=mask)

    """# filtering """
    img_dil = cv.dilate(mask, np.ones((7, 7), np.uint8))

    img_ero = cv.erode(img_dil, np.ones((5, 5), np.uint8))



    """# counting pumpkins """
    conts, hierarchy = cv.findContours(img_ero, cv.RETR_LIST ,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, conts, -1, (255, 0, 0), 1)

    number_of_pumpkins = 0
    centers = []
    for cont in conts:
        M = cv.moments(cont)
        x0 = int(M['m10']/M['m00'])
        y0 = int(M['m01']/M['m00'])
        centers.append((x0, y0))
        # if cv.contourArea(cont) > 900:
        #     number_of_pumpkins += 2
        # else:
        number_of_pumpkins += 1

    return number_of_pumpkins

if __name__ == "__main__" :
    
    path = os.path.dirname(__file__)
    file = os.path.join(path,"../othomosaics/pumkin_filed.tif")

    last_tile = False
    mean_color = np.array([225.69843634, 128.99106478, 176.34921817])
    standard_deviation = np.array([17.55660703, 10.69080414, 9.59233129])
    lower_pumpkin = np.maximum(mean_color - 2 * standard_deviation, 0)
    upper_pumpkin = np.maximum(mean_color + 2 * standard_deviation, 255)

    number_of_pumpkins = 0

    img_full = tile_maneger(file,1000,1000,0)

    while not last_tile:
        [tile,last_tile] = img_full.get_next_tile()

        img_array = np.transpose(tile, (1, 2, 0))
        img = cv.cvtColor(img_array,cv.COLOR_RGB2BGR)

        number_of_pumpkins += count_pumkins(img)

    print("num of pumpkins")
    print(number_of_pumpkins)

