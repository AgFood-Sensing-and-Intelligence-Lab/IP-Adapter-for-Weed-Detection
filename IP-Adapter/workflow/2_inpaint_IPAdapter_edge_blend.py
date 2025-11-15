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

def compute_iou_inner(box_inner, box1):
    # Calculate intersection areas
    y1 = np.maximum(box_inner[0], box1[0])
    y2 = np.minimum(box_inner[2], box1[2])
    x1 = np.maximum(box_inner[1], box1[1])
    x2 = np.minimum(box_inner[3], box1[3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    box_area = (box_inner[2]-box_inner[0])*(box_inner[3]-box_inner[1])
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    union = box_area + box1_area - intersection
    iou = intersection / box_area
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
                                  one_im_labels, one_im_bboxes_xyx1y1, dst_dir, mask_ann_only=False):
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

    merged_ann_info = merge_annotations(bg_ann_info, masked_ann_info, mask_ann_only)
    save_xml(last_im_name_based+'_pasted.xml', image_record='', label=merged_ann_info['one_im_labels'],
            bbox=merged_ann_info['one_im_bboxes_xyx1y1'], save_dir=dst_dir, 
            width=merged_ann_info['im_w'], height=merged_ann_info['im_h'], channel=3)

def find_actual_bbox_for_generated_bbox(generated_bbox):
    bbox, output_masked, output_binary = color_classify_one_box_green(generated_bbox)
    x,y,w,h=bbox
    generated_bbox_cropped = generated_bbox[y:y+h,x:x+w]
    margin_xyx1y1 = [x,y,w,h]
    return margin_xyx1y1, generated_bbox_cropped

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

def edge_blend_inserted_region(background, inserted_region, top_left, padding, blur_strength=5,):
    # Get the dimensions of the inserted region
    h, w = inserted_region.shape[:2]
    
    # Create a mask for the edges (edges will have values between 0 and 1)
    mask = np.ones((h, w), dtype=np.float32)
    
    # Define the region where blending should happen (the edges)
    edge_thickness = padding  # Adjust thickness for how much of the edge will be blended
    
    # Create a radial gradient mask (smooth transition at the edges)
    for y in range(h):
        for x in range(w):
            # Calculate the distance from the border (edges)
            dist_to_edge = min(x, w - x, y, h - y)  # Distance to the nearest edge
            # Normalize the distance and apply a smooth transition
            if dist_to_edge < edge_thickness:
                mask[y, x] = dist_to_edge / edge_thickness
    
    # Apply Gaussian blur to smooth the mask if desired (optional)
    mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)

    # Extract the region from the background where the inserted image will go
    background_patch = background[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    
    # Perform blending only on the edge region
    blended_region = inserted_region * mask[..., None] + background_patch * (1 - mask[..., None])

    # Place the blended region back into the background
    background[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] = blended_region.astype(np.uint8)

    return background

YOLO_mode = None
def detection_model_based_annotation(detection_model_path, masked_im, generated_bbox, mask_ann, instance_ann):
    import sys
    sys.path.insert(0, r'Detection\ultralytics')
    from ultralytics import YOLO
    do_not_find = True
    do_refine = False
    disgard = False
    instance_ann_refined = []
    global YOLO_mode
    if YOLO_mode is None:
        YOLO_mode = YOLO(detection_model_path)
    # 0.05/0.1/0.25
    res = YOLO_mode(masked_im, conf=0.05, iou=0.5)
    predict_bboxes = res[0].boxes.xyxy
    predict_conf_list = res[0].boxes.conf
    predict_cls_list = res[0].boxes.cls

    predict_bboxes = predict_bboxes.cpu().numpy()
    predict_conf_list = predict_conf_list.cpu().numpy()
    predict_cls_list = predict_cls_list.cpu().numpy()
    max_iou = 0
    target_bbox = None
    target_bbox_idx = None
    for i_predict, predict_bbox in enumerate(predict_bboxes):
        iou = compute_iou(instance_ann, predict_bbox)
        if max_iou < iou:
            target_bbox = predict_bbox
            max_iou = iou
            target_bbox_idx = i_predict
    instance_ann_refined = []
    if target_bbox is not None:
        x_mask, y_mask, w_mask, h_mask = mask_ann
        iou_close_to_1 = compute_iou_inner(target_bbox, [x_mask, y_mask, x_mask+w_mask, y_mask+h_mask])
        if iou_close_to_1 > 0.99:
            do_not_find = False
            instance_ann_refined = [int(round(x)) for x in target_bbox]
            if predict_conf_list[target_bbox_idx] >= 0.25 and instance_ann_refined is not None and max_iou > 0.25:
                do_refine = True
            if predict_conf_list[target_bbox_idx] < 0.1:
                disgard = True
    return do_not_find, disgard, do_refine, instance_ann_refined

def green_based_annotation(generated_bbox, mask_bbox, padding, im_w, im_h):
    disgard = False
    x, y, w, h = mask_bbox
    x_new, y_new, x1_new, y1_new = adjust_mask_bbox(x, y, x+w, y+h, padding, im_w, im_h)
    pad_x = x_new-x
    pad_y = y_new-y
    generated_bbox_depad = generated_bbox[pad_y:pad_y+y1_new-y_new, pad_x:pad_x+x1_new-x_new]

    margin_xywh, generated_bbox_cropped = find_actual_bbox_for_generated_bbox(generated_bbox_depad)
    min_size_t= 30
    if margin_xywh[2]*margin_xywh[3] < 0.2*w*h:
        disgard = True
    if margin_xywh[2] < min_size_t or margin_xywh[3] < min_size_t:
        disgard = True
    x_crop = x + margin_xywh[0]
    y_crop = y + margin_xywh[1]
    # fix bug
    x1_crop = x_crop + margin_xywh[2]
    y1_crop = y_crop + margin_xywh[3]

    # masked_im[y:y1, x:x1]=generated_bbox
    # masked_im[y:y1, x:x1]=generated_bbox_cropped
    x_crop_pad = x_crop+padding
    y_crop_pad = y_crop+padding
    if x_crop <= padding:
        x_crop_pad = x_crop
    if y_crop <= padding:
        y_crop_pad = y_crop

    x1_crop_pad = x1_crop+padding
    x1_crop_pad = min(x1_crop_pad, im_w-1)

    y1_crop_pad = y1_crop+padding
    y1_crop_pad = min(y1_crop_pad, im_h-1)
    instance_bbox = [x_crop_pad, y_crop_pad, x1_crop_pad, y1_crop_pad]
    return disgard, generated_bbox_depad, instance_bbox

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

def paste_masks_v2():
    """ 
    85:15
    """
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
    # suffix ='_res512_script'
    suffix ='_res512_script_bioclip'
    # suffix ='_res512_script_v20_seed1'
    use_partial_skus = False
    if use_partial_skus:
        suffix+='_partial_skus'

    # seed_used = 10
    # seed_suffix= f'_seed{seed_used}'

    mask_ann_only = False

    root_dir = r'StableDiffusion\test\generated_10_species'
    data_name = 'train2017_split_1'
    print('data_name:', data_name)

    if data_name == 'train2017_split_1':
        detection_model_path = r'Detection\ultralytics\runs\detect\yolo11l_pretrain_ultralytic_48epochs_1024_train_train2017_split_1_same_im_1024_val_val2017_split_1_20250309_125542\train\weights\best.pt'
    if data_name == 'train2017_split_2':
        detection_model_path = r'Detection\ultralytics\runs\detect\yolo11l_pretrain_ultralytic_48epochs_1024_train_train2017_split_2_same_im_1024_val_val2017_split_2_20250315_194604\train\weights\best.pt'
    if data_name == 'train2017_split_3':
        detection_model_path = r'Detection\ultralytics\runs\detect\yolo11l_pretrain_ultralytic_48epochs_1024_train_train2017_split_3_same_im_1024_val_val2017_split_3_20250317_101744\train\weights\best.pt'

    origin_ann_dir = join(root_dir, data_name)
    masked_im_dir = join(root_dir, f'{data_name}_with_masks')

    generated_bbox_dir = join(root_dir,f'{data_name}_generated_bbox{suffix}')
    dst_dir = join(root_dir,f'{data_name}_generated_im{suffix}_edge_blend_all')

    original_bbox_dir = join(root_dir, f'{data_name}_original_bbox')

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
    bbox_cnt = 0
    find_target =False
    do_not_find_cnt = 0
    disgard_cnt_yolo = 0
    disgard_cnt_color = 0
    for generated_bbox_path in tqdm(generated_bbox_paths):
        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based_tmp, rest_part = file_name_based.split('.png_')
        # if '20210628_iPhoneSE_YL_52_with_masks' != im_name_based_tmp:
        #     continue
        im_name_based = im_name_based_tmp
        if last_im_name_based is not None:
            if last_im_name_based != im_name_based:
                save_path = os.path.join(dst_dir, last_im_name_based+'_pasted.jpg')
                cv2.imwrite(save_path, masked_im)

                if generate_xml:
                    generate_xml_for_merged_image(masked_im, last_im_name_based, origin_ann_dir, masked_im_dir, 
                                                    one_im_labels, one_im_bboxes_xyx1y1, dst_dir, mask_ann_only=mask_ann_only)
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
        try:
            x = int(x)
            y = int(y)
        except Exception as e:
            print(e)
            continue

        original_bbox_path = os.path.join(original_bbox_dir, bbox_name_based+'.jpg')
        if not os.path.exists(original_bbox_path):
            print('original_bbox_path not exist:', bbox_name_based) 
            continue
        generated_bbox = cv2.imread(generated_bbox_path)
        box_h, box_w = generated_bbox.shape[:2]
        masked_im_tmp = edge_blend_inserted_region(masked_im.copy(), generated_bbox, (x, y), padding, blur_strength=5)

        mask_bbox = [x, y, box_w, box_h]
        disgard, generated_bbox_depad, instance_bbox = green_based_annotation(generated_bbox, mask_bbox, padding, im_w, im_h)
        if disgard:
            # continue
            disgard_cnt_color += 1
        else:
            do_not_find_yolo, disgard_yolo, do_refine_yolo, instance_ann_refined = detection_model_based_annotation(detection_model_path, masked_im_tmp, generated_bbox, mask_bbox, instance_bbox)
            if disgard_yolo:
                disgard_cnt_yolo += 1
            if do_not_find_yolo:
                do_not_find_cnt +=1
            if do_refine_yolo:
                instance_bbox = instance_ann_refined
        # if not disgard and not disgard_yolo and not do_not_find_yolo:
        masked_im = masked_im_tmp

        x_crop_pad, y_crop_pad, x1_crop_pad, y1_crop_pad = instance_bbox
        # if do_not_find:
        #     x_crop_pad, y_crop_pad, x1_crop_pad, y1_crop_pad
        #     x_depad, y_depad, x1_depad, y1_depad = generated_bbox_depad
        #     if x_crop_pad > padding and x_crop_pad == x_depad:

        one_im_bboxes_xyx1y1.append([x_crop_pad, y_crop_pad, x1_crop_pad, y1_crop_pad])
        sku_name = bbox_name_based.split('_')[1]
        one_im_labels.append(sku_name)
        bbox_cnt += 1

    save_path = os.path.join(dst_dir, im_name_based+'_pasted.jpg')
    # cv2.imwrite(save_path, masked_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(save_path, masked_im)

    if generate_xml:
        generate_xml_for_merged_image(masked_im, last_im_name_based, origin_ann_dir, masked_im_dir, 
                                        one_im_labels, one_im_bboxes_xyx1y1, dst_dir, mask_ann_only=mask_ann_only)
    one_im_bboxes_xyx1y1 = []
    one_im_labels = []
    print('bbox_cnt:', bbox_cnt)
    print('disgard_cnt_color:', disgard_cnt_color)
    print('disgard_cnt_yolo:', disgard_cnt_yolo)
    print('do_not_find_cnt:', do_not_find_cnt)
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
