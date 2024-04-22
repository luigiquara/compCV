'''
Create the cropped version of GQA, as used in CGQA.

Get the original GQA dataset and, following the information present in json files from CGQA datasets, crop the images to create a classification dataset.
The scripts saves the cropped images using the convention followed by PyTorch ImageFolder 
'''

import json
from multiprocessing import Pool
import os
import time
from tqdm import tqdm
from typing import List, Tuple

from PIL import Image

def crop_and_resize(img_path: str, cropping_coords: List[int], new_size: Tuple[int] = (98,98)) -> Image:
    '''Preprocess an image from GQA to match images from CGQA

    Load image, crop the object and resize.
    The cropped_coords are represented as (y,x,width,height)-tuple
    '''

    img = Image.open(img_path)

    # from (y,x,width,height) to (left, top, right, bottom), as Image.crop() requires
    cropping_coords = (cropping_coords[1], cropping_coords[0], cropping_coords[1]+cropping_coords[3], cropping_coords[0]+cropping_coords[2])
    new_img = img.crop(cropping_coords)

    new_img = new_img.resize(new_size)
    return new_img

def _flatten(xss: List[List]):
    ''' Flatten a list of lists'''
    return [x for xs in xss for x in xs]

def _save_image(img: Image.Image, name: str, label: str, save_path: str):
    '''Save to disk an image

    It follows the PyTorch ImageFolder convention:
    Files are saved as: path/classname/xyz.jpg
    '''
    
    img.save(save_path + label + '/' + name + '.jpg')

def pipeline(file_json: str, save_path: str):
    ''' The complete pipeline for a set of images

    Loads info from CGQA json file, crops the images accordingly and saves them to file, following PyTorch ImageFolder convention
    '''

    # load image info from CGQA dataset
    with open(file_json) as f: img_info = json.load(f)

    # take only the 'object' information
    # it contains the GQA image name, the cropping coords, the label, ...
    # then, go from List[List[Dict]] to List[Dict]
    # each element in img_info is a single object from an image
    #img_info = _flatten([x['objects'] for x in img_info])

    names, bbs, labels, filenames = [], [], [], []
    for comp_img in img_info:
        filename, bb, label = zip(*[(obj['imageName'], obj['boundingBox'], obj['objName']) for obj in comp_img['objects']])

        names.extend([comp_img['newImageName'].split('/')[-1]]*2)
        bbs.extend(bb)
        labels.extend(label)
        filenames.extend(filename)

    # crop the images
    base_path = '/disk3/lquara/GQA/images/'
    #names, bbs, labels = zip(*[(i['imageName'], i['boundingBox'], i['objName']) for i in img_info])
    gqa_paths = [base_path + filename + '.jpg' for filename in filenames]

    print('Starting the cropping')
    start = time.time()
    with Pool(100) as p:
        imgs = p.starmap(crop_and_resize, list(zip(gqa_paths, bbs)))
    print(f'{len(imgs)} images cropped in {time.time()-start} seconds')

    # save cropped images to file
    for img, name, label in tqdm(zip(imgs, names, labels), desc='Saving images'):
        _save_image(img, name, label, save_path)

if __name__ == '__main__':
    img_path = '/disk3/lquara/CGQA/GQA_100'
    train_json_path = os.path.join(img_path, 'continual', 'train', 'train.json')
    val_json_path = os.path.join(img_path, 'continual', 'val', 'val.json')
    test_json_path = os.path.join(img_path, 'continual', 'test', 'test.json')
    
    base_path = '/disk4/lquarantiello/CR_GQA/'
    pipeline(train_json_path, base_path+'train/')
    pipeline(val_json_path, base_path+'val/')
    pipeline(test_json_path, base_path+'test/')