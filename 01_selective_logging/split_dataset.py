import os
import pandas as pd
import numpy as np

from glob import glob

PATH_BASE = '01_selective_logging/data'
PATH_TOTAL_DATASET = '01_selective_logging/data/logging_v2'

# get dataset
list_images = list(glob(f'{PATH_TOTAL_DATASET}/*.tif'))
list_images_df = pd.DataFrame({'path': list_images})



train, validate, test = np.split(
    list_images_df.sample(frac=1), 
    [int(.6*len(list_images_df)), int(.8*len(list_images_df))])

dataset_items = {
    'train': train.values.flatten().tolist(),
    'val': validate.values.flatten().tolist(),
    'test': test.values.flatten().tolist()
}

# mv items
for k, v in dataset_items.items():

    filename = v.split('/')[-1]

    if not os.path.isdir(f'{PATH_BASE}/{k}'):
        os.mkdir(os.path.abspath(f'{PATH_BASE}/{k}'))


    output = os.path.abspath(f'{PATH_BASE}/{k}/{filename}')

    os.system(f'mv {v} {output}')

