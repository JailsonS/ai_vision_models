import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms



PATH_BASE = '01_selective_logging/data'
PATH_TOTAL_DATASET = '01_selective_logging/data/logging_v2'
TYPE = 'train'




def apply_augmentation(image_path, output_path):
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
        transforms.RandomRotation(45),    
        transforms.ToTensor()  
    ])
    augmented_image = transform(image)
    augmented_image.save(output_path)


list_images = glob(f'{PATH_TOTAL_DATASET}/*.tif')


for v in list_images:
    folder_path = os.path.abspath(f'{PATH_BASE}/{TYPE}/{v}')
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    for image_path in v:
        filename = os.path.basename(image_path)
        output_path = os.path.join(folder_path, filename)
        apply_augmentation(image_path, output_path)
