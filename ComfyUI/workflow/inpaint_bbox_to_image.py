import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from tqdm import tqdm
import time
import json
from os.path import join
from PIL import Image, ImageOps
import numpy as np
import traceback


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()
from nodes import NODE_CLASS_MAPPINGS


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


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


use_sku_num_rebalance = True
use_sku_size_match = False

checkpointloadersimple_15 = None
ipadaptermodelloader_40 = None
clipvisionloader_41 = None
bioclipclipvisionloader_253 = None
target_clip_mode = None

use_BioCLIP = True
if use_BioCLIP:
    suffix = '_bioclip'
else:
    suffix = ''
stable_diffusion_path = "v1-5-pruned-emaonly.safetensors"

clip_path = "SD1.5\model.safetensors"
bioclip_path = r"D:\BoyangDeng\StableDiffusionWeedV2\bioclip\model\open_clip_pytorch_model.bin"
bbox_padding = 10
bbox_target_size = 512

split_num = 3

if use_BioCLIP:
    ipadapter_path = f"ip_adapter_plus_sd15_10_species_res512_70k_split{split_num}_bioclip.bin"
else:
    ipadapter_path = f"ip_adapter_plus_sd15_10_species_res512_70k_split{split_num}.bin"

root_dir = r'D:\BoyangDeng\StableDiffusion\test\generated_10_species'
mask_dir = join(root_dir, f"train2017_split_{split_num}_with_masks_v10")
bbox_dir = join(root_dir, f"train2017_split_{split_num}_original_bbox")
save_dir = join(root_dir, f"train2017_split_{split_num}_generated_bbox_res{bbox_target_size}_script{suffix}")

box_num_per_sku_path = join(root_dir, f"box_num_per_sku_train2017_split_{split_num}.json")
assert os.path.exists(box_num_per_sku_path)
with open(box_num_per_sku_path, 'r') as f:
    box_num_per_sku_info = json.load(f)

box_size_per_sku_path = join(root_dir, f"box_size_per_sku_train2017_split_{split_num}.json")
assert os.path.exists(box_size_per_sku_path)
with open(box_size_per_sku_path, 'r') as f:
    box_size_per_sku_info = json.load(f)

import_custom_nodes()

loadimagelistfromdir_inspire = NODE_CLASS_MAPPINGS[
    "LoadImageListFromDir //Inspire"
]()
loadimagenumfromdirlist = NODE_CLASS_MAPPINGS["LoadImageNumFromDirList"]()
ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
previewbridge = NODE_CLASS_MAPPINGS["PreviewBridge"]()
imagesizeandbatchsize = NODE_CLASS_MAPPINGS["ImageSizeAndBatchSize"]()
getfilenamefrompath = NODE_CLASS_MAPPINGS["GetFileNameFromPath"]()
cr_integer_to_string = NODE_CLASS_MAPPINGS["CR Integer To String"]()
text_concatenate = NODE_CLASS_MAPPINGS["Text Concatenate"]()
saveimagetocustomdir = NODE_CLASS_MAPPINGS["SaveImageToCustomDir"]()
df_text = NODE_CLASS_MAPPINGS["DF_Text"]()
cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
multimaskssplit = NODE_CLASS_MAPPINGS["MultiMasksSplit"]()
int_ = NODE_CLASS_MAPPINGS["Int-🔬"]()


def get_each_sku_bbox_path(bbox_dir_files):
    each_sku_bbox_path_dict = {}
    for path in bbox_dir_files:
        im_path = os.path.split(path)[1]
        sku_name = im_path.split('_')[1]
        if sku_name not in each_sku_bbox_path_dict:
            each_sku_bbox_path_dict[sku_name] = []
        each_sku_bbox_path_dict[sku_name] += [path]
    return each_sku_bbox_path_dict


def rebalanced_select_sku(each_sku_bbox_path_dict, bbox_num, box_num_per_sku_info):
    selected_bbox_dir_files = []
    sku_names = list(box_num_per_sku_info.keys())
    probs = []
    tot_prob = 0
    for sku in box_num_per_sku_info:
        probs += [1/box_num_per_sku_info[sku]]
    probs = np.array(probs)
    probs = probs/probs.sum()
    selected_by_sparsity = []
    selected_by_random = []
    if bbox_num > 1:
        half_bbox_num = round(bbox_num / 2)
        selected_by_sparsity = np.random.choice(sku_names, half_bbox_num, True, p=probs)
        if bbox_num - half_bbox_num > 0:
            selected_by_random = np.random.choice(sku_names, bbox_num - half_bbox_num, True)
    else:
        selected_by_sparsity = np.random.choice(sku_names, bbox_num, True, p=probs)
    selected_skus = selected_by_sparsity.tolist()
    if len(selected_by_random) > 0:
        selected_skus = selected_skus + selected_by_random.tolist()
    for selected_sku in selected_skus:
        one_sku_paths = each_sku_bbox_path_dict[selected_sku]
        one_bbox_path = random.choice(one_sku_paths)
        selected_bbox_dir_files += [one_bbox_path]
    return selected_bbox_dir_files


def load_images(dir_files):
    images = []
    masks = []
    file_paths = []
    masks = []
    image_count = 0
    for image_path in dir_files:
        if os.path.isdir(image_path) and os.path.ex:
            continue
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        images.append(image)
        file_paths.append(str(image_path))
        image_count += 1

    return (images, masks, file_paths)


def main():
    mask_paths = list_dir(mask_dir, [], '.png')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generated_bbox_paths = list_dir(save_dir, [], '.png')
    mask_path_num = len(mask_paths)
    t0 = time.time()
    generated_im_names = []
    for generated_bbox_path in tqdm(generated_bbox_paths):
        file_name = os.path.basename(generated_bbox_path)
        file_name_based = os.path.splitext(file_name)[0]
        im_name_based, rest_part = file_name_based.split('.png_')
        if im_name_based not in generated_im_names:
            generated_im_names += [im_name_based]

    # int__250 = int_.execute(value=0)
    loadimagenumfromdirlist_252 = loadimagenumfromdirlist.load_images_with_tot_num(
        directory=bbox_dir,
        image_load_cap=1,
        start_index=0,
        load_always=False,
    )
    tot_bbox_count = get_value_at_index(loadimagenumfromdirlist_252, 3)
    bbox_dir_files = get_value_at_index(loadimagenumfromdirlist_252, 4)
    each_sku_bbox_path_dict = get_each_sku_bbox_path(bbox_dir_files)

    for i_im in tqdm(range(0, mask_path_num)):
        try:
            mask_path = mask_paths[i_im]
            mask_name = os.path.split(mask_path)[1]
            mask_name_base = os.path.splitext(mask_name)[0]
            # im_name = mask_name_base.split('_with_masks')[0]
            if mask_name_base in generated_im_names:
                continue
            one_loop(i_im, tot_bbox_count, each_sku_bbox_path_dict)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
    t1 = time.time()
    print("used time (minute):", (t1 - t0) / 60.0)
    pass


def one_loop(im_start_index, tot_bbox_count, each_sku_bbox_path_dict):
    global checkpointloadersimple_15, ipadaptermodelloader_40, clipvisionloader_41, bioclipclipvisionloader_253, target_clip_mode
    with torch.inference_mode():
        if checkpointloadersimple_15 is None:
            checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            checkpointloadersimple_15 = checkpointloadersimple.load_checkpoint(
                ckpt_name=stable_diffusion_path
            )

        df_text_136 = df_text.get_value(
            Text="Please generate a top-down illustration of a field with a realistic depiction of plants under various lighting conditions"
        )

        df_text_134 = df_text.get_value(Text="Plant")

        text_find_and_replace = NODE_CLASS_MAPPINGS["Text Find and Replace"]()
        text_find_and_replace_84 = text_find_and_replace.text_search_and_replace(
            text=get_value_at_index(df_text_136, 0),
            find="xxx",
            replace=get_value_at_index(df_text_134, 0),
        )

        df_text_135 = df_text.get_value(Text="in a field setting")

        text_find_and_replace_100 = text_find_and_replace.text_search_and_replace(
            text=get_value_at_index(text_find_and_replace_84, 0),
            find="yyy",
            replace=get_value_at_index(df_text_135, 0),
        )

        cliptextencode_18 = cliptextencode.encode(
            text=get_value_at_index(text_find_and_replace_100, 0),
            clip=get_value_at_index(checkpointloadersimple_15, 1),
        )

        cliptextencode_19 = cliptextencode.encode(
            text="blur, text, watermark, CGI, Unreal, Airbrushed, Digital ",
            clip=get_value_at_index(checkpointloadersimple_15, 1),
        )

        loadimagenumfromdirlist_284 = loadimagenumfromdirlist.load_images_with_tot_num(
            directory=mask_dir,
            image_load_cap=1,
            start_index=im_start_index,
            load_always=False,
        )

        multimaskssplit_230 = multimaskssplit.multi_masks_split(
            mask=get_value_at_index(loadimagenumfromdirlist_284, 1)[0]
        )

        """
        consider the sparsity of each sku bbox num (percentage)     
        """
        if use_sku_num_rebalance:
            need_bbox_num = get_value_at_index(multimaskssplit_230, 1)
            selected_bbox_dir_files = rebalanced_select_sku(each_sku_bbox_path_dict, need_bbox_num, box_num_per_sku_info)
            loadimagelistfromdir_inspire_186 = load_images(selected_bbox_dir_files)
        else:
            uniformrandomint = NODE_CLASS_MAPPINGS["UniformRandomInt"]()
            uniformrandomint_206 = uniformrandomint.generate(
                min_val=0,
                max_val=tot_bbox_count,
                # seed=random.randint(1, 2**64),
                seed=im_start_index,
            )
            loadimagelistfromdir_inspire_186 = loadimagelistfromdir_inspire.load_images(
                directory=bbox_dir,
                image_load_cap=get_value_at_index(multimaskssplit_230, 1),
                start_index=get_value_at_index(uniformrandomint_206, 0),
                load_always=False,
            )
        """
        consider each sku bbox size (min, max, median)     
        """
        for i_mask in range(0, multimaskssplit_230[1]):
            one_mask = multimaskssplit_230[0][i_mask]
            mask_crop_region = NODE_CLASS_MAPPINGS["Mask Crop Region"]()
            mask_crop_region_48 = mask_crop_region.mask_crop_region(
                padding=bbox_padding,
                region_type="dominant",
                mask=one_mask,
            )

            int__212 = int_.execute(value=0)

            intnumber = NODE_CLASS_MAPPINGS["IntNumber"]()
            intnumber_215 = intnumber.run(number=1, min_value=0, max_value=3, step=1)

            compare_ = NODE_CLASS_MAPPINGS["Compare-🔬"]()
            compare__211 = compare_.compare(
                comparison="a == b",
                a=get_value_at_index(int__212, 0),
                b=get_value_at_index(intnumber_215, 0),
            )

            if_any_return_a_else_b_ = NODE_CLASS_MAPPINGS["If ANY return A else B-🔬"]()
            if_any_return_a_else_b__209 = if_any_return_a_else_b_.return_based_on_bool(
                ANY=get_value_at_index(compare__211, 0),
                IF_TRUE=get_value_at_index(loadimagenumfromdirlist_284, 0),
                IF_FALSE=get_value_at_index(loadimagenumfromdirlist_284, 0),
            )
            selected_im = if_any_return_a_else_b__209[0][0]
            imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()
            imagecrop_52 = imagecrop.execute(
                width=get_value_at_index(mask_crop_region_48, 6),
                height=get_value_at_index(mask_crop_region_48, 7),
                position="top-left",
                x_offset=get_value_at_index(mask_crop_region_48, 3),
                y_offset=get_value_at_index(mask_crop_region_48, 2),
                image=selected_im,
            )

            imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
            imageresize_94 = imageresize.execute(
                width=bbox_target_size,
                height=bbox_target_size,
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=0,
                image=get_value_at_index(imagecrop_52, 0),
            )

            masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            masktoimage_54 = masktoimage.mask_to_image(
                mask=get_value_at_index(mask_crop_region_48, 0)
            )

            imageresize_96 = imageresize.execute(
                width=bbox_target_size,
                height=bbox_target_size,
                interpolation="lanczos",
                method="keep proportion",
                condition="upscale if smaller",
                multiple_of=0,
                image=get_value_at_index(masktoimage_54, 0),
            )

            imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
            imagetomask_56 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(imageresize_96, 0)
            )

            maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
            maskblur_35 = maskblur.execute(
                amount=6.5, device="auto", mask=get_value_at_index(imagetomask_56, 0)
            )

            inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
            inpaintmodelconditioning_20 = inpaintmodelconditioning.encode(
                positive=get_value_at_index(cliptextencode_18, 0),
                negative=get_value_at_index(cliptextencode_19, 0),
                vae=get_value_at_index(checkpointloadersimple_15, 2),
                pixels=get_value_at_index(imageresize_94, 0),
                mask=get_value_at_index(maskblur_35, 0),
            )
            if ipadaptermodelloader_40 is None:
                ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
                ipadaptermodelloader_40 = ipadaptermodelloader.load_ipadapter_model(
                    ipadapter_file=ipadapter_path
                )
            if target_clip_mode is None: 
                if use_BioCLIP:
                    if bioclipclipvisionloader_253 is None:
                        bioclipclipvisionloader = NODE_CLASS_MAPPINGS["BioCLIPCLIPVisionLoader"]()
                        bioclipclipvisionloader_253 = bioclipclipvisionloader.load_clip(
                            directory=bioclip_path
                        )
                        target_clip_mode = bioclipclipvisionloader_253
                else:
                    if clipvisionloader_41 is None:
                        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
                        clipvisionloader_41 = clipvisionloader.load_clip(
                            clip_name=clip_path
                        )
                        target_clip_mode = clipvisionloader_41
                        if len(target_clip_mode) == 1:
                            target_clip_mode = tuple(list(target_clip_mode)+[None])

            ipadapteradvanced_37 = ipadapteradvanced.apply_ipadapter(
                weight=0.8,
                weight_type="linear",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="V only",
                model=get_value_at_index(checkpointloadersimple_15, 0),
                ipadapter=get_value_at_index(ipadaptermodelloader_40, 0),
                # image=get_value_at_index(loadimagelistfromdir_inspire_186, 0),
                image=get_value_at_index(loadimagelistfromdir_inspire_186, 0)[i_mask],
                attn_mask=get_value_at_index(maskblur_35, 0),
                clip_vision=get_value_at_index(target_clip_mode, 0),
                clip_preprocessor=get_value_at_index(target_clip_mode, 1)
            )

            differentialdiffusion_61 = differentialdiffusion.apply(
                model=get_value_at_index(ipadapteradvanced_37, 0)
            )

            ksampler_16 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=2,
                sampler_name="dpmpp_sde",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(differentialdiffusion_61, 0),
                positive=get_value_at_index(inpaintmodelconditioning_20, 0),
                negative=get_value_at_index(inpaintmodelconditioning_20, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_20, 2),
            )

            vaedecode_17 = vaedecode.decode(
                samples=get_value_at_index(ksampler_16, 0),
                vae=get_value_at_index(checkpointloadersimple_15, 2),
            )

            previewbridge_66 = previewbridge.doit(
                image="$66-0",
                block=False,
                images=get_value_at_index(vaedecode_17, 0),
                unique_id=13614593927737123784,
            )

            imagesizeandbatchsize_97 = imagesizeandbatchsize.batch_size(
                image=get_value_at_index(imagecrop_52, 0)
            )

            imageresize_116 = imageresize.execute(
                width=get_value_at_index(imagesizeandbatchsize_97, 0),
                height=get_value_at_index(imagesizeandbatchsize_97, 1),
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=0,
                image=get_value_at_index(previewbridge_66, 0),
            )

            getfilenamefrompath_301 = getfilenamefrompath.get_filename(
                path=get_value_at_index(loadimagenumfromdirlist_284, 2)[0]
            )

            getfilenamefrompath_303 = getfilenamefrompath.get_filename(
                path=get_value_at_index(loadimagelistfromdir_inspire_186, 2)[i_mask]
            )

            cr_integer_to_string_240 = cr_integer_to_string.convert(
                int_=get_value_at_index(mask_crop_region_48, 3)
            )

            cr_integer_to_string_239 = cr_integer_to_string.convert(
                int_=get_value_at_index(mask_crop_region_48, 2)
            )

            text_concatenate_241 = text_concatenate.text_concatenate(
                delimiter="_xy_",
                clean_whitespace="true",
                text_a=get_value_at_index(cr_integer_to_string_240, 0),
                text_b=get_value_at_index(cr_integer_to_string_239, 0),
            )

            text_concatenate_238 = text_concatenate.text_concatenate(
                delimiter="_",
                clean_whitespace="true",
                text_a=get_value_at_index(getfilenamefrompath_301, 0),
                text_b=get_value_at_index(getfilenamefrompath_303, 0),
                text_c=get_value_at_index(text_concatenate_241, 0),
            )

            saveimagetocustomdir_305 = saveimagetocustomdir.save_images(
                directory=save_dir,
                filename_prefix=get_value_at_index(text_concatenate_238, 0),
                metadata="disable",
                images=get_value_at_index(imageresize_116, 0),
            )


if __name__ == "__main__":
    main()
