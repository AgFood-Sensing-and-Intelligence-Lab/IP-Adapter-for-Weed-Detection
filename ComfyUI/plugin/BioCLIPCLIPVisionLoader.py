from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import open_clip

if __name__ !='__main__':
    import folder_paths
    import comfy.model_management

class BioCLIPCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        # return {"required": { "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
        #                      }}
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
        }
    # RETURN_TYPES = ("CLIP_VISION",)
    RETURN_TYPES = ("CLIP_VISION", "CLIP_PREPROCESSER")
    FUNCTION = "load_clip"

    CATEGORY = "loaders"

    # def load_clip(self, clip_name):
    def load_clip(self, directory):
        # clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        # clip_vision = comfy.clip_vision.load(clip_path)
        device = comfy.model_management.get_torch_device()

        image_encoder_path = directory
        preprocess_train, preprocess_val = None, None
        image_encoder, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained=image_encoder_path)
        image_encoder.eval()  

        # image_encoder = CLIPVisionModelWithProjection.from_pretrained(directory)
        # image_encoder.requires_grad_(False)

        image_encoder.to(device)
        return (image_encoder, preprocess_val)
 

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "BioCLIPCLIPVisionLoader": BioCLIPCLIPVisionLoader
}
 
import torch
import math
import struct
import safetensors.torch
import numpy as np
from PIL import Image
import logging
import itertools


def main():
    image_dir = r'D:\BoyangDeng\StableDiffusion\ComfyUI\models\clip_vision\bioclip\model'
    model = BioCLIPCLIPVisionLoader()
    splitted_masks = model.load_clip(image_dir)


if __name__ =='__main__':
    main()