"""
batch_size=16
im_size=800
"""
import config as cfg
from datetime import datetime
from ultralytics import YOLO, nn
import yaml
from os.path import join
import shutil
import platform
import os
import torch
import time
import sys

cfg.is_val = False

current_datetime = datetime.now()

sys_name = platform.system()

target_name = cfg.target_name

single_cls = False
max_det = cfg.max_det

config_file_dir = r'ultralytics\yolo\data\datasets'
model_dir_pretrained = r'ultralytics\pretrained'
save_period = -1

if target_name == 'weed_stable_diffusion_IPAdapter':
    data_cfg = join(config_file_dir, 'weed_stable_diffusion_IPAdapter.yaml')
    workers = 0

with open(data_cfg, 'r', encoding='UTF-8') as file:
    data_info = yaml.safe_load(file)
data_train = data_info['train']
data_val = data_info['val']

model_name_list = None
resume = False
is_finetunning = False

if target_name == 'weed_stable_diffusion_IPAdapter':
    lr0s = [0.01, ]
    lrfs = [0.01, ]
    warmup_epochs = [3, ]
    warmup_bias_lrs = [0.1, ]
    optimizers = ['SGD']
    model_names = ['yolo11l']
    epochs = [48]
    close_mosaic = min(int(epochs[-1] * 0.2), 10)
    imgsz = 1024
    max_det = 1000
    batch = 4

model_dirs = {}
print('cfg.train_roi:', cfg.train_roi)
run_idx = 0
max_idx = len(model_names)
use_same_hyper = True
while run_idx < max_idx:
    if use_same_hyper:
        hyper_index = 0
    else:
        hyper_index = run_idx

    lr0 = lr0s[hyper_index]
    lrf = lrfs[hyper_index]
    warmup_epoch = warmup_epochs[hyper_index]
    warmup_bias_lr = warmup_bias_lrs[hyper_index]
    optimizer = optimizers[hyper_index]
    epoch = epochs[hyper_index]

    print('lr0:', lr0)
    print('lrf:', lrf)
    print('warmup_epoch:', warmup_epoch)
    print('warmup_bias_lr:', warmup_bias_lr)
    print('optimizer:', optimizer)
    print('epoch:', epoch)
    print('batch:', batch)
    print('single_cls:', single_cls)
    print('max_det:', max_det)
    print('imgsz:', imgsz)

    model_name = model_names[run_idx]

    resume = False
    model_dirs[hyper_index] = model_dir_pretrained

    print('resume:', resume)
    print('model_name:', model_name)
    model_dir = model_dirs[hyper_index]

    subfix = ''
    if cfg.is_debug:
        subfix += '_debug'
    if cfg.use_date:
        formatted_time = current_datetime.strftime("%Y%m%d_%H%M%S")
        subfix += '_'+formatted_time
    pretrained_model_name = model_dirs[hyper_index].strip(os.sep).split(os.sep)[-2]
    save_dir = f'./runs/detect/{model_name}_pretrain_{pretrained_model_name[:10]}_{epochs[0]}epochs_{imgsz}_train_{data_train}_val_{data_val}'+subfix
    print('save_dir:', save_dir)

    project = save_dir

    t0 = time.time()
    model_path = os.path.join(model_dir, model_name+'.pt')
    print('pretained model_path:', model_path)
    # model = YOLO(cfg_model)  # build a new model from scratch
    assert os.path.exists(model_path), model_path
    model = YOLO(model_path)
    # model.model.yaml['channels'] = 4
    # Then rebuild the model with correct channels
    if data_info['channels'] == 4:
        from ultralytics.nn.tasks import parse_model
        from copy import deepcopy
        model.model.model, model.model.save = parse_model(deepcopy(model.model.yaml), ch=4)
        print(f"First conv layer: {model.model.model[0].conv}")
        # If it still shows 3 channels, manually modify it
        if model.model.model[0].conv.in_channels != 4:
            conv = model.model.model[0].conv
            model.model.model[0].conv = nn.Conv2d(
                in_channels=4,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding
            )
            print(f"Modified first conv layer: {model.model.model[0].conv}")
    results = model.train(data=data_cfg,
                          epochs=epoch,
                          imgsz=imgsz,
                          rect=cfg.rect,
                          workers=workers,
                          lr0=lr0, lrf=lrf,
                          warmup_bias_lr=warmup_bias_lr, warmup_epochs=warmup_epoch,
                          optimizer=optimizer, batch=batch,
                          close_mosaic=close_mosaic,
                          save_dir=save_dir,
                          project=project,
                          single_cls=single_cls,
                          max_det=max_det,
                          seed=0,
                          resume=resume,
                          save_period=save_period,
                          )
    t1 = time.time()
    print("used time (minute):", (t1 - t0) / 60.0)
    del model
    torch.cuda.empty_cache()
    run_idx += 1

    # cfg.notify_by_email(model_name)
