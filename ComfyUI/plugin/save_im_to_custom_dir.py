from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import time
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import os
from comfy.cli_args import args
import json


class SaveImageToCustomDir:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"images": ("IMAGE", ),
                 "directory": ("STRING", {"default": ""}),
                 "filename_prefix": ("STRING", {"default": "Mixlab"}),
                 "metadata": (["disable", "enable"],),
                 },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/output"

    def save_images(self, images, directory, filename_prefix="Mixlab", metadata="disable", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        filename = filename_prefix
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, directory, images[0].shape[1], images[0].shape[0])
        counter = 1
        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if (not args.disable_metadata) and (metadata == "enable"):
                print('##enable_metadata')
                from PIL.PngImagePlugin import PngInfo
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            if not os.path.exists(directory):
                os.mkdir(directory)
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(directory, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "directory": directory,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "SaveImageToCustomDir": SaveImageToCustomDir,
}


def main():
    image_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\Eclipta_shuffle'
    model = SaveImageToCustomDir()
    splitted_masks = model.load_images(image_dir)


if __name__ == '__main__':
    main()
