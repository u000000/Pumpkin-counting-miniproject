import cv2 as cv
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import Window,transform
from rasterio import DatasetReader
import os
from typing import Tuple,List
from math import sqrt
from enum import Enum



class tile_maneger:

    def __init__(self,file_path, tile_width, tile_height, overlap) -> None:
        self.file :DatasetReader = rasterio.open(file_path)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap = overlap
        self.ncols, self.nrows = self.file.width, self.file.height
        self.xstep = self.tile_width - overlap
        self.ystep = self.tile_height - overlap
        self.x = -overlap
        self.y = -overlap
        print(f"width:{self.ncols}  height:{self.nrows}")
    
    def get_grid_shape(self) -> int:
        tiles_in_x = int(self.ncols / self.xstep)
        if self.ncols % self.xstep != 0:
            tiles_in_x = tiles_in_x + 1
        tiles_in_y = int(self.nrows / self.ystep)
        if self.nrows % self.ystep != 0:
            tiles_in_y = tiles_in_y + 1    
        return (tiles_in_x,tiles_in_y)
    
    def get_total_tile_count(self) -> int:
        [tiles_in_x,tiles_in_y] = self.get_grid_shape  
        return tiles_in_x*tiles_in_y

    def get_next_tile(self) -> Tuple[any,bool,List[int]]:
        is_last = False
        if (self.y > self.nrows):
                raise "no more tiles"

        window = Window.from_slices(slice(self.y, self.y+self.tile_height),slice(self.x,self.x+self.tile_width))
        data = self.file.read(window=window,boundless=True)

        tile = [self.x/self.xstep,self.y/self.ystep]

        if (self.x + self.xstep < self.ncols):
            self.x = self.x + self.xstep
        else:
            self.x = -self.overlap
            self.y = self.y + self.ystep


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

class place_enum(Enum):
    center = 0
    top_left = 1
    top = 2
    top_right = 3
    left = 4
    right = 5
    buttom_left = 6
    buttom = 7
    buttom_right = 8
    


def count_pumkins(img,tileXY,overlap) -> int:

    mean_color = np.array([225.69843634, 128.99106478, 176.34921817])

    img = cv.cvtColor(img,cv.COLOR_BGR2Lab)
    # Masking
    # mask = cv.inRange(img, lower_pumpkin, upper_pumpkin)
    mask = inEclidianDist(img, mean_color, 30)

    # Mask on original img
    # img_masked = cv.bitwise_and(img, img, mask=mask)

    """# filtering """
    img_ero = cv.erode(mask, np.ones((2, 2), np.uint8))

    img_dil = cv.dilate(img_ero, np.ones((7, 7), np.uint8))
 

    # cv.imshow("img",cv.cvtColor(img,cv.COLOR_Lab2BGR))
    # cv.resizeWindow('img', 600,600)
    # cv.imshow("mask",mask)
    # cv.resizeWindow('mask', 600,600)
    # cv.imshow("img_dil",img_dil)
    # cv.resizeWindow('img_dil', 600,600)
    # cv.imshow("img_ero",img_ero)
    # cv.resizeWindow('img_ero', 600,600)
    # cv.waitKey(0)

    """# counting pumpkins """
    conts, hierarchy = cv.findContours(img_dil, cv.RETR_LIST ,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, conts, -1, (255, 0, 0), 1)

    number_of_pumpkins = 0
    centers = []
    for cont in conts:
        M = cv.moments(cont)
        x0 = int(M['m10']/M['m00'])
        y0 = int(M['m01']/M['m00'])
        centers.append((x0, y0))
        if cv.contourArea(cont) > 900:
            number_of_pumpkins += 2
        else:
            number_of_pumpkins += 1

    return number_of_pumpkins

if __name__ == "__main__" :
    
    path = os.path.dirname(__file__)
    file = os.path.join(path,"../othomosaics/pumkin_filed.tif")
    tileXY = 1000
    overlap = 20

    last_tile = False

    number_of_pumpkins = 0

    img_full = tile_maneger(file,tileXY,tileXY,overlap)

    while not last_tile:
        [tile,last_tile] = img_full.get_next_tile()

        img_array = np.transpose(tile, (1, 2, 0))
        img = cv.cvtColor(img_array,cv.COLOR_RGB2BGR)

        number_of_pumpkins += count_pumkins(img)

    print("num of pumpkins")
    print(number_of_pumpkins)

