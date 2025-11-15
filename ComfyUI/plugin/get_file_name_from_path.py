import os


class GetFileNameFromPath:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename", "dir_part")
    FUNCTION = "get_filename"
    CATEGORY = "Tools"

    def get_filename(self, path):
        dir_part, filename = os.path.split(path)
        return (filename, dir_part)

NODE_CLASS_MAPPINGS = {
    "GetFileNameFromPath": GetFileNameFromPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetFileNameFromPath": "Get FileName From Path"
}

def main():
    image_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\Eclipta_shuffle'
    model = GetFileNameFromPath()
    splitted_masks = model.load_images(image_dir)


if __name__ =='__main__':
    main()