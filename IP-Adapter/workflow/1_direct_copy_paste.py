import cv2
import os
from PIL import Image
from tqdm import tqdm
import random
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from xml_for_generated_im import read_annotation, save_xml, merge_annotations

def list_dir(path, list_name, extension, return_names=False):
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

def adjust_mask_bbox(x, y, x1, y1, padding, im_w, im_h):
    if x > padding:
        x = min(x+padding, im_w-1)
    if y > padding:
        y = min(y+padding, im_h-1)
    if x1 < im_w-padding:
        x1 = max(x1-padding, 0)
    if y1 < im_h-padding:
        y1 = max(y1-padding, 0)
    return x, y, x1, y1

def compute_iou(box, box1):
    # Calculate intersection areas
    y1 = np.maximum(box[0], box1[0])
    y2 = np.minimum(box[2], box1[2])
    x1 = np.maximum(box[1], box1[1])
    x2 = np.minimum(box[3], box1[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box[2]-box[0])*(box[3]-box[1])
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    union = box_area + box1_area - intersection
    iou = intersection / union
    return iou


def match_masked_annotation(masked_ann_info, one_im_labels, one_im_bboxes_xyx1y1):
    # assert len(one_im_labels) == len(masked_ann_info['one_im_labels'])
    if len(one_im_labels) != len(masked_ann_info['one_im_labels']):
        # print('len(one_im_labels):', len(one_im_labels))
        # print('len(masked_ann_info[one_im_labels])', len(masked_ann_info['one_im_labels']))
        pass
    # instance_num = len(masked_ann_info['one_im_labels'])
    # one_im_labels = one_im_labels[instance_num:]

    masked_ann_info['one_im_labels'] = one_im_labels.copy()
    im_w = masked_ann_info['im_w'] 
    im_h = masked_ann_info['im_h'] 
    one_im_bboxes_xyx1y1_ref = masked_ann_info['one_im_bboxes_xyx1y1'].copy()
    # for i_box_ref, xyx1y1_ref in enumerate(one_im_bboxes_xyx1y1_ref):
    #     for i_box, xyx1y1 in enumerate(one_im_bboxes_xyx1y1):
    #         max_iou = 0
    #         x,y,x1,y1 = xyx1y1_ref
    #         x= max(x,0)
    #         y= max(y,0)
    #         x1= min(im_w-1,x1)
    #         y1= min(im_h-1,y1)
    #         xyx1y1_ref_new = [x,y,x1,y1]
    #         iou = compute_iou(xyx1y1, xyx1y1_ref_new)
    #         if iou > max_iou:
    #             max_iou = iou
    #             # masked_ann_info['one_im_bboxes_xyx1y1'][i_box_ref] = xyx1y1_ref_new
    #             masked_ann_info['one_im_bboxes_xyx1y1'][i_box_ref] = xyx1y1
    #             masked_ann_info['one_im_labels'] [i_box_ref] = one_im_labels[i_box]
    masked_ann_info['one_im_bboxes_xyx1y1'] = one_im_bboxes_xyx1y1
    masked_ann_info['one_im_labels'] = one_im_labels
    return masked_ann_info

def generate_xml_for_merged_image(masked_im, last_im_name_based, origin_ann_dir, masked_im_dir, 
                                  one_im_labels, one_im_bboxes_xyx1y1, dst_dir):
    # below code may have issue
    # should consider the padding=10 
    # the masked images have been resized from original images, so xml file should alos be adjusted
    im_h, im_w = masked_im.shape[:2]
    original_im_name = last_im_name_based.split('_with_masks')[0]
    bg_im_ann_path = os.path.join(origin_ann_dir, original_im_name +'.xml')
    bg_ann_info = read_annotation(bg_im_ann_path)

    masked_im_ann_path = os.path.join(masked_im_dir, last_im_name_based +'.xml')
    masked_ann_info = read_annotation(masked_im_ann_path)
    masked_ann_info = match_masked_annotation(masked_ann_info, one_im_labels, one_im_bboxes_xyx1y1)

    merged_ann_info = merge_annotations(bg_ann_info, masked_ann_info)
    save_xml(last_im_name_based+'_pasted.xml', image_record='', label=merged_ann_info['one_im_labels'],
            bbox=merged_ann_info['one_im_bboxes_xyx1y1'], save_dir=dst_dir, 
            width=merged_ann_info['im_w'], height=merged_ann_info['im_h'], channel=3)

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

def copy_paste_bbox(generated_bbox, masked_im, original_bbox_path, padding, im_w, im_h, top_left):
    x,y=top_left
    h_g, w_g = generated_bbox.shape[:2]
    original_bbox = cv2.imread(original_bbox_path)
    h_o, w_o = original_bbox.shape[:2]
    x, y, x1, y1 = adjust_mask_bbox(x, y, x+w_g, y+h_g, padding, im_w, im_h)
    target_size_w=x1-x
    target_size_h=y1-y
    original_bbox = cv2.resize(original_bbox, (target_size_w, target_size_h))
    masked_im[y:y1, x:x1]=original_bbox
    adjust_ann = [x, y, x1, y1]
    return masked_im, adjust_ann

def main():
    generate_xml =True
    sku_name = r'10species'

    particular_cats = [
        'Lambsquarters',
        'PalmerAmaranth',
        'Waterhemp',
        'MorningGlory',
        'Purslane',
        'Goosegrass',
        'Carpetweed',
        'SpottedSpurge',
        'Ragweed',
        'Eclipta',
    ]
    # suffix ='_res256_bioclip'
    suffix ='_res512_script'
    use_partial_skus = False
    if use_partial_skus:
        suffix+='_partial_skus'
    root_dir = r'StableDiffusion\test\generated_10_species'

    data_name = 'train2017_split_3'
    origin_ann_dir = join(root_dir, data_name)
    masked_im_dir = join(root_dir, f'{data_name}_with_masks_v10')

    generated_bbox_dir = join(root_dir,f'{data_name}_generated_bbox{suffix}')

    dst_dir = join(root_dir,f'{data_name}_same_im_copy_paste{suffix}')

    original_bbox_dir = join(root_dir, f'{data_name}_original_bbox')

    if not os.path.exists(generated_bbox_dir):
        os.mkdir(generated_bbox_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    generated_bbox_paths = list_dir(generated_bbox_dir, [], '.png')

    original_bbox_paths = list_dir(original_bbox_dir, [], '.jpg')

    # count_im_num(generated_bbox_paths)

    masked_im = None
    last_im_name_based = None
    merged_ann_info = {}
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    im_w = 0
    im_h = 0
    padding=10
    cnt = 0

    find_target = False

    for generated_bbox_path in tqdm(generated_bbox_paths):
        cnt+=1

        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based, rest_part = file_name_based.split('.png_')

        # if '20230804_HTRC_iPhone12_WY_106_with_masks' == im_name_based:
        #     find_target = True
        # if not find_target:
        #     continue
        if last_im_name_based is not None:
            if last_im_name_based != im_name_based:
                save_path = os.path.join(dst_dir, last_im_name_based+'_pasted.jpg')
                # cv2.imwrite(save_path, masked_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(save_path, masked_im)

                if generate_xml:
                    generate_xml_for_merged_image(masked_im, last_im_name_based, origin_ann_dir, masked_im_dir, 
                                                    one_im_labels, one_im_bboxes_xyx1y1, dst_dir)
                one_im_bboxes_xyx1y1 = []
                one_im_labels = []

                im_path = os.path.join(masked_im_dir, im_name_based+'.png')
                masked_im = cv2.imread(im_path)
                last_im_name_based = im_name_based
        else:
            im_path = os.path.join(masked_im_dir, im_name_based+'.png')
            masked_im = cv2.imread(im_path)
            last_im_name_based = im_name_based
        
        im_h, im_w = masked_im.shape[:2]
        
        bbox_name_based, rest_part = rest_part.split('.jpg_')
        sku_name = bbox_name_based.split('_')[1]
        if sku_name not in particular_cats:
            continue
        if '_00001_' not in rest_part:
            continue
        xy = rest_part.split('_00001_')[0]
        x, y = xy.split('_xy_')
        x = int(x)
        y = int(y)
        original_bbox_path = os.path.join(original_bbox_dir, bbox_name_based+'.jpg')
        if not os.path.exists(original_bbox_path):
            print('original_bbox_path not exist:', bbox_name_based) 
            continue
        generated_bbox = cv2.imread(generated_bbox_path)

        top_left = (x,y)
        masked_im, adjust_ann = copy_paste_bbox(generated_bbox, masked_im, original_bbox_path, padding, im_w, im_h, top_left)
        x, y, x1, y1 = adjust_ann
        
        one_im_bboxes_xyx1y1.append([x, y, x1, y1])
        sku_name = bbox_name_based.split('_')[1]
        one_im_labels.append(sku_name)

    save_path = os.path.join(dst_dir, im_name_based+'_pasted.jpg')
    # cv2.imwrite(save_path, masked_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(save_path, masked_im)

    if generate_xml:
        generate_xml_for_merged_image(masked_im, last_im_name_based, origin_ann_dir, masked_im_dir, 
                                        one_im_labels, one_im_bboxes_xyx1y1, dst_dir)
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    pass

if __name__ =='__main__':
    main()
