import cv2 as cv
import rasterio
from rasterio.plot import show
from rasterio.windows import Window
import os


path = os.path.dirname(__file__)

file = os.path.join(path,"../othomosaics/pumkin_filed.tif")

def get_tiles(file, tile_width, tile_height, overlap):
    nols, nrows = file['width'], file['height']
    xstep = tile_width - overlap
    ystep = tile_height - overlap
    for x in range(0, nols, xstep):
        if x + tile_width > nols:
            x = nols - tile_width
        for y in range(0, nrows, ystep):
            if y + tile_height > nrows:
                y = nrows - tile_height
            window = window.Window(x, y, tile_width, tile_height)
            transform = window.transform(window, file.transform)
            yield window, transform



with rasterio.open(file) as O_file:

    tile_x = O_file.width
    tile_y = O_file.height
    overlap = 50
    metadata = O_file.meta.copy()


    col_start = 0
    row_start = 0
    col2 = O_file.width/2
    row2 = O_file.height/2
    col_end = O_file.width
    row_end = O_file.height   

   #not done
    window = Window.from_slices(slice(row2,row_end), slice(col2,col_end))

    data = O_file.read(window=window)
    show(data)
 



