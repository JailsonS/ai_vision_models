import rasterio
from glob import glob
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.transform import from_bounds
import os

'''
    Config
'''

BASE_PATH = '/home/jailson/Imazon/dl_applications/source/02_patches'
OUTPUT_PATH = f'{BASE_PATH}/mosaics'
YEARS = [
    #'1985',
    #'1986',
    #'1987',
    #'1990',
    #'1993',
    #'1994',
    #'1995',
    #'1996',
    '2000',
    #'2004',
    #'2011',
    #'2016'
]

'''
    Implementation
'''


os.makedirs(OUTPUT_PATH, exist_ok=True)


for year in YEARS:
    
    input_year_path = f'{BASE_PATH}/data/examples_{year}-*'
    list_images = [rasterio.open(x) for x in glob(input_year_path)]
    
    mosaic, output_transform = merge(list_images, method='first', nodata=0)

    output_path = f'{OUTPUT_PATH}/mosaic_{year}.tif'



    out_meta = list_images[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output_transform,
        "crs": list_images[0].crs
    })



    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    print(f'Mosaico para o ano {year} salvo em: {output_path}')
