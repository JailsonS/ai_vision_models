
'''
    Import Session
'''

import sys, os

sys.path.append(os.path.abspath('.'))


import ee


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com', project='ee-simex')


bigGridList = [
    "NA-19-Y",
    "NA-19-Z",
    "NA-20-X",
    "NA-22-Z",
    "NB-22-Y",
    "SA-19-X",
    "SA-19-Z",
    "SA-20-X",
    "SA-20-Y",
    "SA-20-Z",
    "SA-21-Y",
    "SA-22-V",
    "SA-22-X",
    "SA-22-Y",
    "SA-23-V",
    "SA-23-X",
    "SA-23-Z",
    "SB-18-X",
    "SB-20-V",
    "SB-20-X",
    "SB-20-Y",
    "SB-20-Z",
    "SB-21-V",
    "SC-18-X",
    "SC-22-Z",
    "SD-20-V",
    "SE-20-X",
    "SE-21-V"
]

cartasName = [

    'SB-20-Z-C',
    'SB-20-Z-D',
    'SB-20-Z-B',
    'SB-20-X-C',
    'SB-20-X-D',
    'SB-21-V-C',
    'SB-21-V-A',
    'SB-21-V-B',
    'SB-21-V-D',
    'SA-21-Y-D',
    'SA-21-Y-B',
    'SA-21-Y-C',
    'SA-21-Y-A',
    'SA-22-V-C',
    'SA-22-Y-C',
    'SA-22-Y-A',
    'SA-22-V-D',
    'SA-22-Y-B',
    'SA-22-Y-D',
    'SA-22-X-C',
    'SA-22-X-D',
    'SA-23-V-C',
    'SC-22-Z-C',
    'SC-22-Z-A',
    'SC-22-Z-B',
    'SC-22-Z-D',
    'SE-21-V-D',
    'SE-21-V-B',
    'SE-21-V-A',
    'SE-20-X-B',
    'SD-20-V-B',
    'SD-20-V-A'


]

grids = ee.FeatureCollection('users/mapbiomas_c1/bigGrids_paises');
leapGrids = grids.filter(ee.Filter.inList('grid_name', ee.List(bigGridList)))


amz = ee.Image("projects/mapbiomas-workspace/AUXILIAR/biomas-2019-raster")

FABDEM = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
FABDEM = FABDEM.mosaic().rename('fabdem').toInt16().updateMask(amz.eq(1))


for name in cartasName:

    grids = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cartas")\
                 .filter(ee.Filter.eq('grid_name', name))
    
    dem = FABDEM

    print(f'exporting image: {name}')

    task = ee.batch.Export.image.toDrive(
        image=dem,
        description= name + '-DEM-BR-AMAZ-FABDEM',
        folder='SAGA-4', 
        region=grids.geometry().buffer(10000).bounds(),
        scale=30,
        maxPixels=1e+13
    )

    task.start()
