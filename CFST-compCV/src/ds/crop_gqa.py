'''
Create the cropped version of GQA, as used in CGQA.

Get the original GQA dataset and, following the information present in json files from CGQA datasets, crop the images to create a classification dataset.
The scripts saves the cropped images using the convention followed by PyTorch ImageFolder 
'''

import json
import os
from typing import List, Tuple

from PIL import Image

def _crop_image(img_path: str, cropping_coords: Tuple[int,int,int,int]) -> Image:
    ''' Crop a single object from GQA image

    Parameters:
    ----------
    '''

def save_cropped_dataset(tr_json: str, val_json: str, test_json: str):
    # load info about the images
    # e.g. name of the file, cropping coords
    with open(tr_json) as f: train_img_info = json.load(f)
    with open(val_json) as f: val_img_info = json.load(f)
    with open(test_json) as f: test_img_info = json.load(f)



if __name__ == '__main__':
    img_path = '/disk3/lquara/CGQA/GQA_100'
    train_json_path = os.path.join(img_path, 'continual', 'train', 'train.json')
    val_json_path = os.path.join(img_path, 'continual', 'val', 'val.json')
    test_json_path = os.path.join(img_path, 'continual', 'test', 'test.json')
    
    save_cropped_dataset(train_json_path, val_json_path, test_json_path)