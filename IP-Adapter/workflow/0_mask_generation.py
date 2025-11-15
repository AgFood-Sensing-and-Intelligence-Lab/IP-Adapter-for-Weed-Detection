import cv2
import os
from PIL import Image
from tqdm import tqdm
import random
import traceback
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from xml_for_generated_im import read_annotation


def generate_gaussian_mask(im_size, anchor):
    # Parameters for the 2D Gaussian
    x_center, y_center, anchor_size_x, anchor_size_y = anchor
    loc_center = [x_center, y_center]
    anchor_size = [anchor_size_x, anchor_size_y]
    loc_x = loc_center[0]
    loc_y = loc_center[1]
    x_center = im_size[0]//2
    y_center = im_size[1]//2
    mu_x = loc_x-x_center
    mu_y = loc_y-y_center
    sigma_x, sigma_y = anchor_size[0]//2/3, anchor_size[1]//2/3  # Standard deviations
    # sigma_x, sigma_y = anchor_size[0]//2, anchor_size[1]//2  # Standard deviations
    rho = 0.0  # Correlation coefficient

    # Create a grid of (x, y) points
    x = np.linspace(-im_size[0]//2, im_size[0]//2, im_size[0])
    y = np.linspace(-im_size[1]//2, im_size[1]//2, im_size[1])
    X, Y = np.meshgrid(x, y)

    # Calculate the Z values of the 2D Gaussian using the PDF formula
    Z = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))) * np.exp(
        -1 / (2 * (1 - rho ** 2)) * (
            (X - mu_x) ** 2 / sigma_x ** 2 +
            (Y - mu_y) ** 2 / sigma_y ** 2 -
            (2 * rho * (X - mu_x) * (Y - mu_y)) / (sigma_x * sigma_y)
        )
    )
    return Z


def postporocess_Z_for_merge(Z):
    Z = Z[:, :, None]
    Z = Z*255.0
    Z[Z > 0] = Z[Z > 0]/(Z[Z > 0].max()-Z[Z > 0].min())*255
    # Z[Z>0]=255
    Z = 255-Z
    Z[Z > 250] = 255
    Z[Z < 250] = 0
    Z = Z.astype(np.uint8)
    return Z


def save_xml(image_name, image_record, size_total, label, bbox, save_dir, width=2048, height=2048):
    from xml.etree.ElementTree import Element, SubElement, tostring

    from xml.dom.minidom import parseString

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    if image_record is not None:
        node_folder = SubElement(node_root, 'image_record')
        node_folder.text = image_record

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')

    node_size_total = SubElement(node_size, 'size_total')
    node_size_total.text = '%s' % size_total

    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    cnt = 0
    max_size = max(width, height)
    for x, y, x1, y1 in bbox:
        if x1 <= x | y1 <= y | x < 0 | y < 0 | x1 > max_size | y1 > max_size:
            print(x, y, x1, y1)
            continue
        left, top, right, bottom = x, y, x1, y1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label[cnt]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
        cnt += 1

    xml = tostring(node_root)
    dom = parseString(xml)
    image_name_base = image_name.split('.')[0]
    save_xml_path = os.path.join(save_dir, image_name_base+'.xml')
    with open(save_xml_path, 'wb') as f:
        f.write(xml)


def generate_annotation(save_path, Z, boxes, im_size, save_dir):
    valid_sku_im_score_dict = {}
    im_info_dict = {}
    im_ext = '.png'
    im_path = save_path
    im_dir = os.path.dirname(im_path)
    file_name = os.path.basename(im_path)
    base_name = file_name.split(im_ext)[0]
    xml_name = base_name+'.xml'
    image_name = base_name+im_ext

    # im = cv2.imread(im_path)
    width, height = im_size
    one_im_labels = []
    one_im_bboxes_xyx1y1 = []
    for idx_xywh in boxes:
        x, y, w, h, sku_name = idx_xywh[:5]
        one_im_labels += [sku_name]
        xmin = round((x-w/2))
        xmax = round((x+w/2))
        ymin = round((y-h/2))
        ymax = round((y+h/2))
        one_im_bboxes_xyx1y1 += [[xmin, ymin, xmax, ymax]]

    image_name = base_name
    save_xml(image_name, image_record='', size_total=width*height, label=one_im_labels,
             bbox=one_im_bboxes_xyx1y1, save_dir=im_dir, width=width, height=height)


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
        print(traceback.format_exc())
    return list_name


def compute_iou(box1, box2):
    box_ref = box1
    tb = min(box1[2], box2[2]) - \
        max(box1[0], box2[0])
    lr = min(box1[3], box2[3]) - \
        max(box1[1], box2[1])
    inter = 0 if tb < 0 or lr < 0 else tb * lr
    w1 = box1[2]-box1[0]
    h1 = box1[3]-box1[1]
    w2 = box2[2]-box2[0]
    h2 = box2[3]-box2[1]
    # return inter / (w1*h1+w2*h2 - inter)
    return inter / (w1*h1)


def get_box_size_info_in_one_image(ann_info):
    one_im_box_size_info = {}
    im_w = ann_info['im_w']
    im_h = ann_info['im_h']
    one_im_labels = ann_info['one_im_labels']
    one_im_bboxes_xyx1y1 = ann_info['one_im_bboxes_xyx1y1']
    box_sizes = []
    one_im_box_size_info['box_sizes'] = []
    for box in one_im_bboxes_xyx1y1:
        [x1, y1, x2, y2] = box
        box_w = x2-x1
        box_h = y2-y1
        box_size = (box_w*box_h)**0.5
        # box_sizes+=[box_size]
        box_sizes += [box_w, box_h]
    one_im_box_size_info['box_sizes'] = box_sizes
    box_sizes = np.array(box_sizes)
    one_im_box_size_info['box_size_mean'] = np.mean(box_sizes)
    one_im_box_size_info['box_size_std'] = np.std(box_sizes)
    one_im_box_size_info['box_size_median'] = np.median(box_sizes)
    one_im_box_size_info['box_size_min'] = np.min(box_sizes)
    one_im_box_size_info['box_size_max'] = np.max(box_sizes)
    return one_im_box_size_info


def is_mask_overlap_existed_anchors(anchor, existed_anchors, iou_t=0.25):
    [x_center, y_center, anchor_size_x, anchor_size_y] = anchor
    box_anchor = [x_center-anchor_size_x//2, y_center-anchor_size_y//2, x_center+anchor_size_x//2, y_center+anchor_size_y//2]
    is_overlapped = False
    iou = 0.0
    for box in existed_anchors:
        [x_center, y_center, anchor_size_x, anchor_size_y] = box
        box_anchor_existed = [x_center-anchor_size_x//2, y_center-anchor_size_y//2, x_center+anchor_size_x//2, y_center+anchor_size_y//2]
        iou = compute_iou(box_anchor_existed, box_anchor)
        if iou > iou_t:
            is_overlapped = True
            break
    return iou, is_overlapped


def is_mask_overlap_existed_bboxes(anchor, origin_ann_info, iou_t=0.25):
    [x_center, y_center, anchor_size_x, anchor_size_y] = anchor
    im_w = origin_ann_info['im_w']
    im_h = origin_ann_info['im_h']
    one_im_labels = origin_ann_info['one_im_labels']
    one_im_bboxes_xyx1y1 = origin_ann_info['one_im_bboxes_xyx1y1']
    is_overlapped = False
    iou = 0.0
    for box_ref in one_im_bboxes_xyx1y1:
        box_anchor = [x_center-anchor_size_x//2, y_center-anchor_size_y//2, x_center+anchor_size_x//2, y_center+anchor_size_y//2]
        iou = compute_iou(box_ref, box_anchor)
        if iou > iou_t:
            is_overlapped = True
            break
    return iou, is_overlapped


def resize_ann_for_resized_im(ann_info, origin_size, new_size):
    bg_im_w_origin, bg_im_h_origin = origin_size
    bg_im_w, bg_im_h = new_size
    w_ratio = bg_im_w/bg_im_w_origin
    h_ratio = bg_im_h/bg_im_h_origin
    ann_info_resized = {}
    ann_info_resized['im_w'] = bg_im_w
    ann_info_resized['im_h'] = bg_im_h
    ann_info_resized['one_im_labels'] = ann_info['one_im_labels']
    ann_info_resized['one_im_bboxes_xyx1y1'] = []

    one_im_bboxes_xyx1y1 = ann_info['one_im_bboxes_xyx1y1']
    for box in one_im_bboxes_xyx1y1:
        xmin, ymin, xmax, ymax = box
        xmin *= w_ratio
        xmax *= w_ratio
        ymin *= h_ratio
        ymax *= h_ratio
        box_resized = [xmin, ymin, xmax, ymax]
        ann_info_resized['one_im_bboxes_xyx1y1'] += [box_resized]
    return ann_info_resized


def generate_valid_anchors(anchor_layer_num, last_layer_size, bg_im_w, bg_im_h, 
                           origin_ann_info_resized, valid_anchors, ignore_edge_mask, iou_t=0, im_name=''):
    valid_anchors = {}
    anchor_tot_num = 0
    for i_layer in range(1, anchor_layer_num+1):
        anchor_size_x = last_layer_size
        anchor_size_y = last_layer_size
        last_layer_size = last_layer_size*2

        x_stride_num_max = int(np.ceil(bg_im_w/anchor_size_x))
        x_gap = max(anchor_size_x*x_stride_num_max - bg_im_w, 0)

        y_stride_num_max = int(np.ceil(bg_im_h/anchor_size_y))
        y_gap = max(anchor_size_y*y_stride_num_max - bg_im_h, 0)

        start_x = anchor_size_x//2 - x_gap//2
        start_y = anchor_size_y//2 - y_gap//2

        stride_x = anchor_size_x//2
        stride_y = anchor_size_y//2
        if x_stride_num_max > 5:
            stride_x = stride_x*2
        if y_stride_num_max > 5:
            stride_y = stride_y*2

        x_stride_num = int(np.ceil(bg_im_w/stride_x))
        y_stride_num = int(np.ceil(bg_im_h/stride_y))
        # start_x = anchor_size_x//2
        # start_y = anchor_size_y//2

        # x_stride_num = round((bg_im_w - start_x)/stride_x)
        # y_stride_num = round((bg_im_h - start_y)/stride_y)
        for i_stride_y in range(0, y_stride_num):
            for i_stride_x in range(0, x_stride_num):
                x_center = start_x+stride_x*i_stride_x
                y_center = start_y+stride_y*i_stride_y

                x_center = max(x_center, anchor_size_x//4)
                y_center = max(y_center, anchor_size_x//4)
                x_center = min(x_center, bg_im_w-anchor_size_x//4)
                y_center = min(y_center, bg_im_h-anchor_size_y//4)

                if ignore_edge_mask:
                    margin_ratio_t = 0.25
                    is_out_edge_low = (x_center-anchor_size_x < 0-margin_ratio_t*anchor_size_x) or (y_center-anchor_size_y < 0-margin_ratio_t*anchor_size_y)
                    is_out_edge_high = (x_center+anchor_size_x > bg_im_w+margin_ratio_t*anchor_size_x) or (y_center+anchor_size_y > bg_im_h+margin_ratio_t*anchor_size_y)
                    if is_out_edge_low or is_out_edge_high:
                        continue

                anchor = [x_center, y_center, anchor_size_x, anchor_size_y]
                # add extra margins for bbox?
                # iou_t=0.1 may decrease mAP@50?
                try:
                    iou, is_overlapped = is_mask_overlap_existed_bboxes(anchor, origin_ann_info_resized, iou_t=iou_t)
                    if is_overlapped:
                        continue
                except Exception as e:
                    print(im_name, e)
                    print(traceback.format_exc())
                    continue
                # if random.uniform(0, 1) > 0.2:
                #     continue
                disgard_this_anchor = False
                for i_anchor in valid_anchors:
                    iou, is_overlapped = is_mask_overlap_existed_anchors(anchor, valid_anchors[i_anchor], iou_t=0.0)
                    if is_overlapped:
                        disgard_this_anchor = True
                if disgard_this_anchor:
                    continue
                if i_layer not in valid_anchors:
                    valid_anchors[i_layer] = []

                prob = random.uniform(0, 1)
                if len(valid_anchors[i_layer]) == 0:
                    pass
                elif i_layer == 1 and prob < 0.8:
                    continue
                elif i_layer == 2 and prob < 0.5:
                    continue
                elif prob < 0.2:
                    continue
                valid_anchors[i_layer] += [anchor]
                anchor_tot_num += 1
    return valid_anchors, anchor_tot_num


def mask_generation():
    """ 
    seed= 0
    seed= 10
    seed= 1
    """
    seed= 0

    # Z=generate_gaussian_mask((100,100),(800, 1000), (200,200))
    # max_size_t = 4032
    # max_size_t = 2464
    max_size_t = 1024
    """ 
    mask_per_im_t_max = 10
    mask_per_im_t_max = 4
    mask_per_im_t_max = 20
    """
    mask_per_im_t_max = 4
    mask_per_im_t_min = 1
    """
    mask_per_im_t_no_available = 1
    mask_per_im_t_no_available = 2
    """
    mask_per_im_t_no_available = 1
    im_num_limit_t = -1
    # im_num_limit_t = 100
    """
    anchor_num_each_layer_types = {'less': 2, 'more': 5}
    anchor_num_each_layer_types = {'less': 2, 'more': 6}
    """
    anchor_num_each_layer_types = {'less': 2, 'more': 5}

    spit_num = 1
    print('spit_num:', spit_num)
    only_cnt_no_save = False
    
    
    root_dir = r'StableDiffusion\test\generated_10_species'
    bg_dir = os.path.join(root_dir, f'train2017_split_{spit_num}')


    save_dir = os.path.join(root_dir, f'train2017_split_{spit_num}_with_masks_v20_seed{seed}')

    # ref_dir = os.path.join(root_dir, 'train2017_split_1_with_masks')
    # generated_mask_paths = list_dir(ref_dir, [], '.png', return_names=True)
    generated_mask_paths = []

    bg_xml_paths = list_dir(bg_dir, [], '.xml')
    random.seed(seed)
    random.shuffle(bg_xml_paths)

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
    sku_num = len(particular_cats)
    im_number = len(bg_xml_paths)
    num_im_for_each_sku = im_number/sku_num

    target_skus = particular_cats
    generate_im_num = 0
    generate_instance_num = 0

    im_without_generation_list = []
    for i_im, bg_xml_path in tqdm(enumerate(bg_xml_paths)):
        one_im_generate_instance_num = 0
        # if i_im < 300:
        #     continue
        if im_num_limit_t > 0:
            if i_im >= im_num_limit_t:
                break
        sku_idx = int(np.floor(i_im / num_im_for_each_sku))
        target_sku = 'Weed'

        save_dir_one_sku = save_dir

        if not os.path.exists(save_dir_one_sku):
            os.mkdir(save_dir_one_sku)
        im_name = os.path.basename(bg_xml_path).split('.xml')[0]
        bg_im_path = os.path.join(bg_dir, im_name+'.jpg')
        if im_name +'_with_masks.png' in generated_mask_paths:
            continue

        ann_path = os.path.join(bg_dir, im_name+'.xml')
        origin_ann_info = read_annotation(ann_path)

        bg_im = cv2.imread(bg_im_path)
        bg_im_h_origin, bg_im_w_origin, channel_num = bg_im.shape[:3]

        # if max(bg_im_h_origin, bg_im_w_origin) > max_size_t:
        if bg_im_h_origin > bg_im_w_origin:
            bg_im_w = round(max_size_t*bg_im_w_origin/bg_im_h_origin)
            bg_im_h = max_size_t
        else:
            bg_im_h = round(max_size_t*bg_im_h_origin/bg_im_w_origin)
            bg_im_w = max_size_t
        bg_im = cv2.resize(bg_im, (bg_im_w, bg_im_h))
        bg_im = bg_im[:, :, :3]

        max_size_im = max(bg_im_w, bg_im_h)

        origin_ann_info_resized = resize_ann_for_resized_im(origin_ann_info, (bg_im_w_origin, bg_im_h_origin), (bg_im_w, bg_im_h))
        # actual_gt_num = len(origin_ann_info_resized['one_im_bboxes_xyx1y1']) + 1
        actual_gt_num = len(origin_ann_info_resized['one_im_bboxes_xyx1y1'])
        mask_per_im_t = max(mask_per_im_t_min, actual_gt_num)
        mask_per_im_t = min(mask_per_im_t, mask_per_im_t_max)

        one_im_box_size_info = get_box_size_info_in_one_image(origin_ann_info_resized)

        # size_min_t = 50
        size_min_t = 64
        im_min_size = min(bg_im_h, bg_im_w)
        large_box_size = im_min_size
        box_size_min = one_im_box_size_info['box_size_median'] - one_im_box_size_info['box_size_std']//2
        box_size_min = max(box_size_min, one_im_box_size_info['box_size_min'])
        box_size_min = round(box_size_min)
        if box_size_min < large_box_size:
            ref_box_size = box_size_min
        else:
            ref_box_size = min(large_box_size, box_size_min/2)

        #  50 200
        # basic_anchor_size = 100
        basic_anchor_size = int(max(ref_box_size//size_min_t*size_min_t, size_min_t))
        # max_size_scale =1.0
        max_size_scale =1.3
        max_size = min((one_im_box_size_info['box_size_max']*max_size_scale//size_min_t+1)*size_min_t, max_size_im)
        # basic_anchor_size_scale = 1.0
        basic_anchor_size_scale = 0.7
        anchor_layer_num = int(np.ceil(np.log2(max_size)-np.log2(basic_anchor_size*basic_anchor_size_scale)))

        if anchor_layer_num == 0:
            anchor_layer_num = 1
            # basic_anchor_size = 50
            basic_anchor_size = size_min_t

        last_layer_size = basic_anchor_size
        valid_anchors = []

        valid_anchors, anchor_tot_num = generate_valid_anchors(anchor_layer_num, last_layer_size, bg_im_w, bg_im_h, 
                                                origin_ann_info_resized, valid_anchors, True, iou_t=0.0, im_name=im_name)
        iou_t = 0.25
        while anchor_tot_num == 0 and iou_t <= 1.0:
            last_layer_size = last_layer_size / 2
            mask_per_im_t = mask_per_im_t_no_available
            valid_anchors, anchor_tot_num = generate_valid_anchors(anchor_layer_num, last_layer_size, bg_im_w, bg_im_h, 
                                                    origin_ann_info_resized, valid_anchors, False, iou_t=iou_t, im_name=im_name)
            iou_t += 0.25
        if anchor_tot_num == 0:
            print(im_name, 'no location to generate masks')
        selected_valid_anchors_tot = []
        used_anchor_num = 0
        anchor_num_all_layers = []
        anchor_ratio_all_layers_dict = {}
        valid_layer_num = 0
        anchor_ratio_all_layers = []
        for i_anchor_layer in valid_anchors:
            anchor_num = len(valid_anchors[i_anchor_layer])
            if anchor_num > 0:
                valid_layer_num += 1
            anchor_num = min(anchor_num, np.ceil(mask_per_im_t/2))
            anchor_num_all_layers.append(anchor_num)
        anchor_num_all_layers = np.array(anchor_num_all_layers)
        tot_anchor_num_all_layers_modified = anchor_num_all_layers.sum() 
        if tot_anchor_num_all_layers_modified >= 1:
            anchor_ratio_all_layers = anchor_num_all_layers/tot_anchor_num_all_layers_modified
        for i_anchor_layer in valid_anchors:
            anchor_ratio_all_layers_dict[i_anchor_layer] = anchor_ratio_all_layers[i_anchor_layer-1]

        for i_anchor_layer in valid_anchors:
            one_layer_valid_anchors = valid_anchors[i_anchor_layer]
            one_layer_valid_anchors = np.array(one_layer_valid_anchors)
            anchor_num_all = len(one_layer_valid_anchors)
            left_anchor_num = anchor_tot_num - anchor_num_all
            anchor_ratio = anchor_ratio_all_layers_dict[i_anchor_layer]
            anchor_num = int(np.ceil(anchor_ratio*mask_per_im_t))

            left_needed_anchor_num = mask_per_im_t - used_anchor_num
            anchor_num_each_layer_t = anchor_num_each_layer_types['more']
            if i_anchor_layer == 1 and len(valid_anchors) > 1:
                if left_anchor_num >= mask_per_im_t:
                    anchor_num_each_layer_t = anchor_num_each_layer_types['less']
                else:
                    anchor_num_each_layer_t = mask_per_im_t - left_anchor_num
            elif used_anchor_num < mask_per_im_t and len(valid_anchors)==i_anchor_layer:
                anchor_num_each_layer_t = left_needed_anchor_num

            if anchor_num == 0:
                continue
            else:
                valid_anchor_num = min(anchor_num_each_layer_t, left_needed_anchor_num)
                valid_anchor_num = min(anchor_num, valid_anchor_num)
                # mask_per_layer = np.random.randint(valid_anchor_num-1, valid_anchor_num+1)
                mask_per_layer = valid_anchor_num

                actual_anchor_num = len(one_layer_valid_anchors)
                mask_per_layer = max(1, min(actual_anchor_num, mask_per_layer))
                if actual_anchor_num > 0:
                    selected_idxes = np.random.choice(actual_anchor_num, mask_per_layer, replace=False)
                    selected_valid_anchors = one_layer_valid_anchors[selected_idxes]
                if len(selected_valid_anchors) > 0:
                    selected_valid_anchors_tot += selected_valid_anchors.tolist()
                    used_anchor_num = len(selected_valid_anchors_tot)
            
        anchor_tot_num_filtered = len(selected_valid_anchors_tot)
        if anchor_tot_num_filtered > mask_per_im_t_max:
            selected_idxes = np.random.choice(anchor_tot_num_filtered, mask_per_im_t_max, replace=False)
            selected_valid_anchors_tot = np.array(selected_valid_anchors_tot)
            selected_valid_anchors_tot = selected_valid_anchors_tot[selected_idxes]

        one_im_generate_instance_num += len(selected_valid_anchors_tot)

        Z_tot = np.ones_like(bg_im[:, :, 0:1]) * 255
        boxes = []
        try:
            for anchor in selected_valid_anchors_tot:
                x_center, y_center, anchor_size_x, anchor_size_y = anchor
                Z = generate_gaussian_mask((bg_im_w, bg_im_h), anchor)
                Z = postporocess_Z_for_merge(Z)
                Z_tot[Z == 0] = Z[Z == 0]
                box = [x_center, y_center, anchor_size_x, anchor_size_y, target_sku]
                boxes += [box]
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            continue

        bg_im_with_mask = np.concatenate([bg_im, Z_tot], 2)
        save_path = os.path.join(save_dir_one_sku, im_name+'_with_masks.png')
        if one_im_generate_instance_num == 0:
            im_without_generation_list += [im_name]
        else:
            generate_im_num += 1
            generate_instance_num += one_im_generate_instance_num
            if only_cnt_no_save:
                continue

            res = cv2.imwrite(save_path, bg_im_with_mask)
            # res = cv2.imwrite(save_path, bg_im_with_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # cv2.imwrite('output_image.png', bg_im_with_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            # may not need, due to the generated bbox's name include the sku name
            res = generate_annotation(save_path, Z_tot, boxes, (bg_im_w, bg_im_h), save_dir_one_sku)

    print('generate_im_num:', generate_im_num)
    print('generate_instance_num:', generate_instance_num)
    print('im_without_generation_list num:', len(im_without_generation_list))
    f_summary = open(os.path.join(root_dir, "summary_no_mask_im_list.json"), 'w')
    import json
    summary_dict = {}
    summary_dict['im_without_mask_list'] = im_without_generation_list
    json.dump(summary_dict, f_summary, indent=4)
    f_summary.close()

def count_im_num():
    # suffix ='_bioclip'
    suffix ='_res256'
    root_dir = r'StableDiffusion\test\generated_10_species'

    data_name = 'train2017_split_3'
    generated_bbox_dir = join(root_dir,f'{data_name}_generated_bbox{suffix}')
    generated_bbox_paths = list_dir(generated_bbox_dir, [], '.png')
    generated_mask_dir = join(root_dir,f'{data_name}_with_masks')
    generated_mask_paths = list_dir(generated_mask_dir, [], '.png')

    data_name = 'train2017_split_3'
    original_im_dir = join(root_dir,f'{data_name}')
    original_im_paths = list_dir(original_im_dir, [], '.jpg')
    print('masked im_num:', len(generated_mask_paths))
    print('original im_num:', len(original_im_paths))
    return 
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
    print('generated masked im_num:', im_num)
    return 

def main():
    mask_generation()
    # count_im_num()
    pass
if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    #     print(traceback.format_exc())
