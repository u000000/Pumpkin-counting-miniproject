import cv2 as cv
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import Window,transform
from rasterio import DatasetReader
import os
from typing import Tuple



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
        # print(f"x:{self.x}  y:{self.y}")

        window = Window.from_slices(slice(self.y, self.y+self.tile_height),slice(self.x,self.x+self.tile_width))
        # window_transform = transform(window, self.file.transform)
        # print(window)
        data = self.file.read(window=window,boundless=True)



        if (self.x + self.xstep < self.ncols):
            self.x = self.x + self.xstep
        else:
            self.x = 0
            self.y = self.y + self.ystep


        if (self.y > self.nrows):
            is_last = True
        
        return data,is_last
    

if __name__ == "__main__" :

    path = os.path.dirname(__file__)

    file = os.path.join(path,"../othomosaics/pumkin_filed.tif")


    img = tile_maneger(file,1000,1000,50)
    print(img.get_total_tile_count())
    for i in range(img.get_total_tile_count()-2):
        [data,v] = img.get_next_tile()
        img_array = np.transpose(data, (1, 2, 0))
        img_array = cv.cvtColor(img_array,cv.COLOR_RGB2BGR)
        cv.imshow("img",img_array)
    [data,v] = img.get_next_tile()
    print(f"x:{img.x}  y:{img.y} shape {img.get_grid_shape()}")
    img_array = np.transpose(data, (1, 2, 0))
    cv.imshow("img",img_array)
    cv.waitKey(0)




