
import rasterio

from glob import glob
from rasterio.merge import merge

'''
    Config
'''


BASE_PATH = '/home/jailson/Imazon/dl_applications/source/02_patches'

OUTPUT_PATH = f'{BASE_PATH}/data'

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

    
for year in YEARS:

	input_year_path = '{}/data/examples_{}-*'.format(BASE_PATH, year)

	list_images = [rasterio.open(x) for x in glob(input_year_path)]
	
	mosaic, output = merge(list_images, method = 'first', nodata = 0)

	print(output)

