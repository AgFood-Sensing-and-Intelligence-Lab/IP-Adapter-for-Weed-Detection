import cv2
import os
from PIL import Image
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from xml_for_generated_im import read_annotation, save_xml, merge_annotations

def list_dir(path, list_name, extension, use_name=False):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                if use_name:
                    list_name.append(file)
                else:
                    list_name.append(file_path)
    try:
        list_name = sorted(list_name, key=lambda k: int(os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name


def adjust_mask_bbox(x, y, x1, y1, padding, im_w, im_h):
    if x > 0:
        x = x+padding
    if y > 0:
        y = y+padding
    if x1 < im_w-1:
        x1 = x1-padding
    if y1 < im_h-1:
        y1 = y1-padding
    return x, y, x1, y1


def count_im_num(generated_bbox_paths):
    im_num = 0
    last_im_name_based = None
    for generated_bbox_path in tqdm(generated_bbox_paths):
        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based, rest_part = file_name_based.split('.png_')
        if last_im_name_based is not None:
            if last_im_name_based != im_name_based:
                im_num += 1
                last_im_name_based = im_name_based
        else:
            last_im_name_based = im_name_based
    print('im_num:', im_num)
    return 

def get_target_im_names_for_IPADapter(ref_dir):
    base_file_names=[]
    # multi instances per image
    file_paths = list_dir(ref_dir, [], '.jpg', use_name=True)
    for file_path in file_paths:
        generated_name = os.path.basename(file_path)

        file_name_base = os.path.splitext(generated_name)[0]
        bg_im_name = file_name_base.split('_with_masks_pasted')[0] 
        base_file_names.append(bg_im_name)
    return base_file_names

def main():
    generate_xml =True
    from os.path import join
    sku_name = r'10species'
    root_dir = r'StableDiffusion\test\generated_10_species'
    target_size = 1024
    data_name = 'train2017_split_1'
    origin_ann_dir = join(root_dir, data_name)
    origin_im_dir = join(root_dir, data_name)
    masked_im_dir = join(root_dir, f'{data_name}_with_masks')

    suffix ='_res256'
    
    ref_dir = join(root_dir,f'{data_name}_generated_im{suffix}')

    base_im_names = get_target_im_names_for_IPADapter(ref_dir)

    dst_dir = join(root_dir,f'{data_name}_same_im_{target_size}')

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_im_names = list_dir(dst_dir,[], '.jpg', use_name=True)

    last_im_name_based = None
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    im_w = 0
    im_h = 0
    cnt = 0

    for base_im_name in tqdm(base_im_names):
        im_name_based = os.path.splitext(base_im_name)[0]
        cnt+=1

        im_path = os.path.join(origin_im_dir, base_im_name+'.jpg')
        mask_im_path = os.path.join(masked_im_dir, base_im_name+'_with_masks.png')

        ann_path = os.path.join(origin_ann_dir, im_name_based+'.xml')
        origin_im = cv2.imread(im_path)
        im_h, im_w = origin_im.shape[:2]
        if im_h > im_w:
            im_h_t = target_size
            im_w_t = round(im_w*im_h_t/im_h)
        else:
            im_w_t = target_size
            im_h_t = round(im_h*im_w_t/im_w)
        # resize_im = cv2.resize(origin_im, (im_w_t, im_h_t))
        resize_im = cv2.imread(mask_im_path)

        save_path = os.path.join(dst_dir, im_name_based+'.jpg')
        if save_path.endswith('.jpg'):
            # cv2.imwrite(save_path, resize_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(save_path, resize_im)
        # else:
        #     # cv2.imwrite('output_image.png', resize_im, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     cv2.imwrite(save_path, resize_im, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        #     tmp_im = cv2.imread(save_path)
        #     os.remove(save_path)
        #     save_path = os.path.join(dst_dir, im_name_based+'.jpg')
        #     cv2.imwrite(save_path, tmp_im)

        ann_info = read_annotation(ann_path)
        if generate_xml:
            bboxes_resized = []
            for bbox in ann_info['one_im_bboxes_xyx1y1']:
                w_ratio = im_w_t / ann_info['im_w']
                h_ratio = im_h_t / ann_info['im_h']
                xmin, ymin, xmax, ymax = bbox
                xmin *= w_ratio
                xmax *= w_ratio
                ymin *= h_ratio
                ymax *= h_ratio
                bbox_resized = [xmin, ymin, xmax, ymax]
                bbox_resized = [round(x) for x in bbox_resized]
                bboxes_resized.append(bbox_resized)
            ann_info['one_im_bboxes_xyx1y1'] = bboxes_resized
            ann_info['im_w'] = im_w_t
            ann_info['im_h'] = im_h_t
            save_xml(im_name_based+'.xml', image_record='', label=ann_info['one_im_labels'],
                    bbox=ann_info['one_im_bboxes_xyx1y1'], save_dir=dst_dir, 
                    width=ann_info['im_w'], height=ann_info['im_h'], channel=3)
    print('cnt:', cnt)
    pass

if __name__ =='__main__':
    main()
