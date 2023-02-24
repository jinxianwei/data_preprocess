import os.path as osp
import os
import mmcv
from tqdm import tqdm

import pandas as pd 
import ast
import mmengine
import numpy as np
import random
from shutil import copy, rmtree

def mk_file(file_path: str):
    if osp.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

## IMAGE FILES
path_folder_images_png = '/root/autodl-tmp/SEM_rawdata/PNG/'

# 1. 对数据集进行划分，并保证随机性可复现
random.seed(0)
split_rate = 0.2  # train:val=8:2

train_root = "/root/autodl-tmp/SEM_div2k/train_HR"
mk_file(train_root)
val_root = "/root/autodl-tmp/SEM_div2k/valid_HR"
mk_file(val_root)

train_list = []
val_list = []

all_sem_imgs = os.listdir(path_folder_images_png)
num = len(all_sem_imgs)

# 随机采样验证集的索引，并将数据copy到train和val目录中
eval_index = random.sample(all_sem_imgs, k=int(num*split_rate))
for png_img in mmengine.track_iter_progress(all_sem_imgs):
    if png_img in eval_index:
        # 将分配至验证集中的文件复制到相应目录
        image_path = osp.join(path_folder_images_png, png_img)
        new_path = val_root
        copy(image_path, new_path)
    else:
        # 将copy至train文件中
        image_path = osp.join(path_folder_images_png, png_img)
        new_path = train_root
        copy(image_path, new_path)