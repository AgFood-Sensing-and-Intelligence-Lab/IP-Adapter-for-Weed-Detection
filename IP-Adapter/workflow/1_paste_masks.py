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

def find_actual_bbox_for_generated_bbox(generated_bbox):
    bbox, output_masked, output_binary = color_classify_one_box_green(generated_bbox)
    x,y,w,h=bbox
    generated_bbox_cropped = generated_bbox[y:y+h,x:x+w]
    margin_xyx1y1 = [x,y,w,h]
    return margin_xyx1y1, generated_bbox_cropped

def paste_masks():
    generate_xml =True
    sku_name = r'10species'
    # suffix ='_bioclip'
    # suffix ='_v0'
    suffix ='_res256'
    
    root_dir = r'StableDiffusion\test\generated_10_species'

    data_name = 'train2017_split_2'
    origin_ann_dir = join(root_dir, data_name)
    masked_im_dir = join(root_dir, f'{data_name}_with_masks_2')

    # generated_bbox_dir = join(root_dir,f'generated_bboxes_{sku_name}{suffix}')
    # dst_dir = join(root_dir,f'generated_im_{sku_name}{suffix}')
    generated_bbox_dir = join(root_dir,f'{data_name}_generated_bbox{suffix}_2')
    
    dst_dir = join(root_dir,f'{data_name}_generated_im{suffix}_2')

    if not os.path.exists(generated_bbox_dir):
        os.mkdir(generated_bbox_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    generated_bbox_paths = list_dir(generated_bbox_dir, [], '.png')

    masked_im = None
    last_im_name_based = None
    merged_ann_info = {}
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    im_w = 0
    im_h = 0
    padding=10
    cnt = 0

    for generated_bbox_path in tqdm(generated_bbox_paths):
        cnt+=1
        # print('cnt:', cnt)
        # if cnt <= 5:
        #     continue
        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based, rest_part = file_name_based.split('.png_')

        # if '20210806_iPhoneSE_YL_188_with_masks' != im_name_based:
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
        if '_00001_' not in rest_part:
            continue
        xy = rest_part.split('_00001_')[0]
        x, y = xy.split('_xy_')
        try:
            x = int(x)
            y = int(y)
        except Exception as e:
            print(e)
            continue
        generated_bbox = cv2.imread(generated_bbox_path)
        h, w = generated_bbox.shape[:2]
        x1=x+w
        y1=y+h
        masked_im[y:y1, x:x1]=generated_bbox

        x, y, x1, y1 = adjust_mask_bbox(x, y, x+w, y+h, padding, im_w, im_h)
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

def color_classify_one_box_green(img_roi):
    import cv2 as cv
    import copy
    lower_color = np.array([30])
    upper_color = np.array([80])

    obj_ = img_roi.copy()

    h = obj_.shape[0]
    w = obj_.shape[1]
    area = h*w

    obj_hsv = cv2.cvtColor(obj_, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(obj_hsv)

    mask_color = cv2.inRange(H, lower_color, upper_color)
    output = cv2.bitwise_and(H, H, mask=mask_color)

    output_masked = copy.deepcopy(output)
    min_size = min(h,w)
    if min_size >= 200:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        iterations = 2
        output = cv.erode(output, kernel, iterations=iterations)
        iterations = 4
        output = cv.dilate(output, kernel, iterations=iterations)
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        iterations = 1
        output = cv.erode(output, kernel, iterations=iterations)
        iterations = 2
        output = cv.dilate(output, kernel, iterations=iterations)
    ret, output_binary = cv2.threshold(output, 1, 255, cv2.THRESH_BINARY)
    res = cv.findContours(image=output_binary, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    # print(len(contours))
    # im_one_conotur = np.zeros_like(diff_im_optimize_open)
    # cv.drawContours(image=im_one_conotur, contours=[contour], contourIdx=-1,
    #                 color=255, thickness=-1, lineType=cv.LINE_AA)
    list_of_pts = []
    for contour in contours:
        for pt in contour:
            list_of_pts.append(pt[0].tolist())
    ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
    ctr = cv2.convexHull(ctr)
    x, y, w, h = cv2.boundingRect(ctr)
    # print('ratio: ', ratio)
    bbox = [x, y, w, h]
    return bbox, output_masked, output_binary


def paste_masks_v2():
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
    dst_dir = join(root_dir,f'{data_name}_generated_im{suffix}')

    if not os.path.exists(generated_bbox_dir):
        os.mkdir(generated_bbox_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    generated_bbox_paths = list_dir(generated_bbox_dir, [], '.png')

    masked_im = None
    last_im_name_based = None
    merged_ann_info = {}
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    im_w = 0
    im_h = 0
    padding=10
    cnt = 0
    find_target =False
    for generated_bbox_path in tqdm(generated_bbox_paths):
        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based, rest_part = file_name_based.split('.png_')

        # if '72_with_masks' == im_name_based:
        #     return
        # if '30_with_masks' == im_name_based:
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
        cnt+=1

        xy = rest_part.split('_00001_')[0]
        x, y = xy.split('_xy_')
        try:
            x = int(x)
            y = int(y)
        except Exception as e:
            print(e)
            continue
        generated_bbox = cv2.imread(generated_bbox_path)
        h, w = generated_bbox.shape[:2]
        x1=x+w
        y1=y+h

        x_new, y_new, x1_new, y1_new = adjust_mask_bbox(x, y, x+w, y+h, padding, im_w, im_h)
        pad_x = x_new-x
        pad_y = y_new-y
        generated_bbox_adjust = generated_bbox[pad_y:pad_y+y1_new-y_new, pad_x:pad_x+x1_new-x_new]

        margin_xywh, generated_bbox_cropped = find_actual_bbox_for_generated_bbox(generated_bbox_adjust)
        min_size_t= 30
        if margin_xywh[2]*margin_xywh[3] < 0.2*w*h:
            continue
        if margin_xywh[2] < min_size_t or margin_xywh[3] < min_size_t:
            continue
        x = x + margin_xywh[0]
        y = y + margin_xywh[1]
        x1 = x + margin_xywh[2]
        y1 = y + margin_xywh[3]

        # masked_im[y:y1, x:x1]=generated_bbox
        masked_im[y:y1, x:x1]=generated_bbox_cropped

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
    print('bbox cnt:', cnt)
    pass

def main():
    # paste_masks()

    """
    specify sku
    from original images instead of masked images
    tighten the bbox
    """
    paste_masks_v2()
    pass

if __name__ =='__main__':
    main()
