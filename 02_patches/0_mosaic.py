from osgeo import gdal
from osgeo import gdalconst
from glob import glob



OUTPUT_PATH = '/content/imazon/mapbiomas/degradation/fragmentation/forest_mosaic/'

YEARS = [str(x) for x in range(1985,2023)]

def merge_mosaic(rasters, output_path, year):

    gdal.BuildVRT(output_path + "forest_mosaic_{}.vrt".format(year), rasters)

    gdal.Translate(
        output_path + "forest_mosaic_{}.tif".format(year),
        output_path + "forest_mosaic_{}.vrt".format(year),
        outputType=gdalconst.GDT_Byte,
        creationOptions=['TFW=NO', 'COMPRESS=LZW', 'TILED=YES', 'COPY_SRC_OVERVIEWS=YES'])
    
    
for year in YEARS:

  mosaic_path = '/content/imazon/mapbiomas/degradation/fragmentation/forest_classification/forest_classification_{}_*.tif'.format(year)

  mosaic = glob(mosaic_path)
  print('Forest mosaic {}'.format(year))

  merge_mosaic(mosaic, OUTPUT_PATH, year)

  print('--')