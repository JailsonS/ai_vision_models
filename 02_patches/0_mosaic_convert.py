from utils.converstion import *



BASE_PATH = '02_patches/data'

YEARS = [
    1985
]


for year in YEARS:

    path_geotif = '{}/mosaic_{}.tif'.format(BASE_PATH, str(year))
    path_netcdf = '{}/netcdf_{}.nc'.format(BASE_PATH, str(year))

    geotiff_to_netcdf(path_geotif, path_netcdf)

