from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import time
import torch
import cv2 as cv
import matplotlib.pyplot as plt
# You can use this node to save full size images through the websocket, the
# images will be sent in exactly the same format as the image previews: as
# binary images on the websocket with a 8 byte header indicating the type
# of binary message (first 4 bytes) and the image format (next 4 bytes).

# Note that no metadata will be put in the images saved with this node.


class MultiMasksSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {"mask": ("MASK", ), },
            "optional": {
                # "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                # "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                # "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("MASK","INT", "INT")
    RETURN_NAMES = ("SPLITTED MASK", "MASK NUMBER", "MASK INDEX")
    OUTPUT_IS_LIST = (True, False, True)

    FUNCTION = "multi_masks_split"

    # OUTPUT_NODE = True

    CATEGORY = "image/mask"

    def multi_masks_split(self, mask):
        splitted_masks = []
        mask_indexes = []

        if isinstance(mask, list):
            mask = mask[0]
        if isinstance(mask, np.ndarray):
            mask_np = mask
        else:
            mask_np = mask.cpu().numpy()
        if mask_np.max() <= 1.0:
            mask_np = mask_np * 255.0
            mask_np = mask_np.astype(np.uint8)
        res = cv.findContours(image=mask_np, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = res
        cnt = 0
        for contour in contours:
            mask_indexes+=[cnt]
            cnt+=1
            bbox = cv.boundingRect(contour)
            (x,y,w,h) = bbox
            empty_mask = np.zeros_like(mask)
            empty_mask[y:y+h, x:x+w] = mask_np[y:y+h, x:x+w]
            empty_mask = empty_mask.astype(np.float32) / 255.0
            empty_mask = torch.from_numpy(empty_mask)[None,]
            splitted_masks.append(empty_mask)
        print(splitted_masks[0].shape)
        mask_number = len(splitted_masks)
        print('mask number:', mask_number)
        return (splitted_masks, mask_number, mask_indexes)


NODE_CLASS_MAPPINGS = {
    "MultiMasksSplit": MultiMasksSplit,
}

def main():
    image_path = r'D:\BoyangDeng\StableDiffusion\test\generated_Purslane\mask_soil_with_weeds_no_overlap\259_with_masks.png'
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    model = MultiMasksSplit()
    splitted_masks = model.multi_masks_split(mask)


if __name__ =='__main__':
    main()