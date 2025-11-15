# IP-Adapter-for-Weed-Detection
This is the official repo for the paper "Image Prompt Adapter-Based Stable Diffusion for Enhanced Multi-Class Weed Generation and Detection"

## Citation
Please consider cite our work if you find this repo is helpful.
```
@article{
  title={Image Prompt Adapter-Based Stable Diffusion for Enhanced Multi-Class Weed Generation and Detection},
  author={Lu, Yuzhen and Deng, Boyang},
  journal={AgriEngineering},
  volume={7},
  pages={389},
  year={2025},
  publisher={MDPI}
}
```

## Contents

1) The ComfyUI code used for developping workflow to generate synthetic images. Modified from https://github.com/comfyanonymous/ComfyUI
2) The Detection code used for training the weed detection model based on https://github.com/ultralytics/ultralytics
3) The ImageQualityMetrics code used for assesssing the FID/IS of synthetic images.
4) The IP-Adapter code used for training the weed generation model.Modified from https://github.com/tencent-ailab/IP-Adapter
