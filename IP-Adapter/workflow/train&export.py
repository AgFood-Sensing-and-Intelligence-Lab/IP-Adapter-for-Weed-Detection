import torch
import os


root_dir = r'StableDiffusion\IP-Adapter\output_models_sd_ip_adapter_MorningGlory\checkpoint-15000'
root_dir = r'StableDiffusion\IP-Adapter\output_models_sd_ip_adapter_PalmerAmaranth\checkpoint-20000'
root_dir = r'StableDiffusion\IP-Adapter\output_models_sd_ip_adapter_MultipleSpecies_10species\checkpoint-120000'
ckpt = os.path.join(root_dir, "pytorch_model.bin")
sd = torch.load(ckpt, map_location="cpu")
clip_vision = {}
image_proj_sd = {}
ip_sd = {}
for k in sd:
    print(k)
    if k.startswith("unet"):
        pass
    elif k.startswith("clipvision"):
        clip_vision[k.replace("clipvision.", "")] = sd[k]
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

# torch.save({"clipvision": clip_vision, "image_proj": image_proj_sd, "ip_adapter": ip_sd},  os.path.join(root_dir, "ip_adapter.bin"))
