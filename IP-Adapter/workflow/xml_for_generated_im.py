import cv2
import os
from PIL import Image
from tqdm import tqdm
import random

import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import ElementTree
from os.path import join


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
        list_name = sorted(list_name, key=lambda k: int(os.path.split(k)[1].split(extension)[0].split('_')[-1]))
    except Exception as e:
        print(e)
    return list_name


def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def if_match(node, kv_map):
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


def get_node_by_keyvalue(nodelist, kv_map):
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


def find_nodes(tree, path):
    return tree.findall(path)


def save_xml(image_name, image_record, label, bbox, save_dir, width=2048, height=2048, channel=3):
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

    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
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
    image_name_base = os.path.splitext(image_name)[0]
    save_xml_path = os.path.join(save_dir, image_name_base+'.xml')
    with open(save_xml_path, 'wb') as f:
        f.write(xml)


def read_annotation(xml_path):
    tree = read_xml(xml_path)
    object_nodes = get_node_by_keyvalue(find_nodes(tree, "object"), {})
    if len(object_nodes) == 0:
        print(xml_path, "no object")
        return
    width_nodes = get_node_by_keyvalue(find_nodes(tree, "size/width"), {})
    height_nodes = get_node_by_keyvalue(find_nodes(tree, "size/height"), {})
    im_w = int(width_nodes[0].text)
    im_h = int(height_nodes[0].text)

    name_nodes = get_node_by_keyvalue(find_nodes(tree, "object/name"), {})
    xmin_nodes = get_node_by_keyvalue(
        find_nodes(tree, "object/bndbox/xmin"), {})
    ymin_nodes = get_node_by_keyvalue(
        find_nodes(tree, "object/bndbox/ymin"), {})
    xmax_nodes = get_node_by_keyvalue(
        find_nodes(tree, "object/bndbox/xmax"), {})
    ymax_nodes = get_node_by_keyvalue(
        find_nodes(tree, "object/bndbox/ymax"), {})
    one_im_labels = []
    one_im_bboxes_xyx1y1 = []
    for index, node in enumerate(object_nodes):
        xmin = int(xmin_nodes[index].text)
        ymin = int(ymin_nodes[index].text)
        xmax = int(xmax_nodes[index].text)
        ymax = int(ymax_nodes[index].text)
        one_label = name_nodes[index].text
        one_im_labels += [one_label]
        one_im_bboxes_xyx1y1.append([xmin, ymin, xmax, ymax])
    ann_info = {}
    ann_info['im_w'] = im_w
    ann_info['im_h'] = im_h
    ann_info['one_im_labels'] = one_im_labels
    ann_info['one_im_bboxes_xyx1y1'] = one_im_bboxes_xyx1y1
    return ann_info


def merge_annotations(bg_ann_info, mask_ann_info, mask_ann_only=False):
    ann_info = {}
    ann_info['im_w'] = mask_ann_info['im_w']
    ann_info['im_h'] = mask_ann_info['im_h']
    if mask_ann_only:
        ann_info['one_im_labels'] = mask_ann_info['one_im_labels']
    else:
        ann_info['one_im_labels'] = bg_ann_info['one_im_labels']+mask_ann_info['one_im_labels']

    bg_bboxes_resized = []
    for bbox in bg_ann_info['one_im_bboxes_xyx1y1']:
        w_ratio = mask_ann_info['im_w'] / bg_ann_info['im_w']
        h_ratio = mask_ann_info['im_h'] / bg_ann_info['im_h']
        xmin, ymin, xmax, ymax = bbox
        xmin *= w_ratio
        xmax *= w_ratio
        ymin *= h_ratio
        ymax *= h_ratio
        bbox_resized = [xmin, ymin, xmax, ymax]
        bbox_resized = [round(x) for x in bbox_resized]
        bg_bboxes_resized.append(bbox_resized)
    if mask_ann_only:
        ann_info['one_im_bboxes_xyx1y1'] = mask_ann_info['one_im_bboxes_xyx1y1']
    else:
        ann_info['one_im_bboxes_xyx1y1'] = bg_bboxes_resized+mask_ann_info['one_im_bboxes_xyx1y1']
    return ann_info


def copy_file(src, dst):
    import shutil
    shutil.copy(src, dst)


def main():
    sku_name = r'PalmerAmaranth'
    root_dir = r'StableDiffusion\test\generated_10_species'
    generated_dir = join(root_dir, f'generated_im_{sku_name}')
    bg_im_dir = join(root_dir, 'bg_weed_images')
    mask_im_dir = join(root_dir, f'mask_soil_with_weeds_no_overlap_{sku_name}')
    generated_bbox_dir = join(root_dir, f'generated_bboxes_{sku_name}')

    dst_dir = generated_dir
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    generated_bbox_names = list_dir(generated_bbox_dir, [], '.jpg', return_names=True)
    # png jpg
    generated_paths = list_dir(generated_dir, [], '.jpg')
    for generated_path in tqdm(generated_paths):
        generated_name = os.path.basename(generated_path)
        file_name_based = os.path.splitext(generated_name)[0]

        # mask_name, others= file_name_based.split('.png_')[:2]
        # instance_name = others.split('.jpg_00001_')[0]
        # bg_im_name = mask_name.split('_size_')[0]

        if '_with_masks_pasted' not in file_name_based:
            continue
        mask_name, others= file_name_based.split('_pasted')[:2]

        bg_im_name = mask_name.split('_with_masks')[0]

        bg_im_ann_name = bg_im_name+'.xml'
        bg_im_path = os.path.join(bg_im_dir, bg_im_name+'.jpg')
        bg_im_ann_path = os.path.join(bg_im_dir, bg_im_ann_name)
        bg_ann_info = read_annotation(bg_im_ann_path)

        mask_ann_path = os.path.join(mask_im_dir, mask_name+'.xml')
        mask_ann_info = read_annotation(mask_ann_path)

        merged_ann_info = merge_annotations(bg_ann_info, mask_ann_info)
        save_xml(generated_name, image_record='', label=merged_ann_info['one_im_labels'],
                 bbox=merged_ann_info['one_im_bboxes_xyx1y1'], save_dir=generated_dir, width=merged_ann_info['im_w'], height=merged_ann_info['im_h'], channel=3)

        # dst_bg_im_path = os.path.join(dst_dir, bg_im_name+'.jpg')
        # dst_bg_im_ann_path = os.path.join(dst_dir, bg_im_ann_name)
        # copy_file(bg_im_path, dst_bg_im_path)
        # copy_file(bg_im_ann_path, dst_bg_im_ann_path)


if __name__ == '__main__':
    main()
