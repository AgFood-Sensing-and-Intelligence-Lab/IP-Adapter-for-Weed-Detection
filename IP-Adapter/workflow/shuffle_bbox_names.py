import cv2
import os
from PIL import Image
from tqdm import tqdm
import random
import shutil

import numpy as np
import matplotlib.pyplot as plt
from xml_for_generated_im import read_annotation

def list_dir(path, list_name, extension, return_names=False):
    import os
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                if return_names:
                    list_name.append(file)
                else:
                    list_name.append(file_path)
    try:
        list_name = sorted(list_name)
    except Exception as e:
        print(e)
    return list_name

def main():
    
    
    box_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_split_3_object_in_box\_MultipleSpecies_10species'
    dst_dir = box_dir+'_shuffle'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    src_im_paths = list_dir(box_dir, [], '.jpg')
    random.seed(0)
    random.shuffle(src_im_paths)
    for i_im, src_im_path in tqdm(enumerate(src_im_paths)):
        bg_im_name = os.path.basename(src_im_path)
        dst_path_im = os.path.join(dst_dir, str(i_im)+'_'+bg_im_name)
        
        # src_dir = os.path.dirname(src_im_path)
        # im_name_base = bg_im_name.split('.jpg')[0]
        # src_ann_path = os.path.join(src_dir, im_name_base+'.xml')
        # dst_path_ann = os.path.join(dst_dir, str(i_im)+'_'+im_name_base+'.xml')

        shutil.copyfile(src_im_path, dst_path_im)
        # shutil.copyfile(src_ann_path, dst_path_ann)


if __name__ == '__main__':
    main()
