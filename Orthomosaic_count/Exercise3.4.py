import cv2 as cv
import rasterio
from rasterio.plot import show
from rasterio.windows import Window

file = "ex.tif"

with rasterio.open(file) as O_file:
    col_start = 0
    row_start = 0
    col2 = O_file.width/2
    row2 = O_file.height/2
    col_end = O_file.width
    row_end = O_file.height   
    window = Window.from_slices((row_start,row2,row_end), (col_start,col2,col_end))

    data = O_file.read(window=window)

 



