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

path_val_HR = "/root/autodl-tmp/SEM_div2k/valid_HR"
path_valid_LR_bicubic_X2 = "/root/autodl-tmp/SEM_div2k/valid_LR_bicubic/X2"
path_valid_LR_bicubic_X3 = "/root/autodl-tmp/SEM_div2k/valid_LR_bicubic/X3"
path_valid_LR_bicubic_X4 = "/root/autodl-tmp/SEM_div2k/valid_LR_bicubic/X4"

path_gtmod12_prefix = "/root/autodl-tmp/SEM_set5_14/GTmod12"
path_set514_LRbicx2_prefix = "/root/autodl-tmp/SEM_set5_14/LRbicx2"
path_set514_LRbicx3_prefix = "/root/autodl-tmp/SEM_set5_14/LRbicx3"
path_set514_LRbicx4_prefix = "/root/autodl-tmp/SEM_set5_14/LRbicx4"

mk_file(path_gtmod12_prefix)
all_val_HR_img = os.listdir(path_val_HR)
for png_img in mmengine.track_iter_progress(all_val_HR_img):
    image_path = osp.join(path_val_HR, png_img)
    new_path = osp.join(path_gtmod12_prefix, png_img)
    copy(image_path, new_path)

mk_file(path_set514_LRbicx2_prefix)
all_X2_img = os.listdir(path_valid_LR_bicubic_X2)
for png_img in mmengine.track_iter_progress(all_X2_img):
    image_path = osp.join(path_valid_LR_bicubic_X2, png_img)
    new_name = png_img[:-6] + png_img[-4:]
    new_path = osp.join(path_set514_LRbicx2_prefix, new_name)
    copy(image_path, new_path)

mk_file(path_set514_LRbicx3_prefix)
all_X3_img = os.listdir(path_valid_LR_bicubic_X3)
for png_img in mmengine.track_iter_progress(all_X3_img):
    image_path = osp.join(path_valid_LR_bicubic_X3, png_img)
    new_name = png_img[:-6] + png_img[-4:]
    new_path = osp.join(path_set514_LRbicx3_prefix, new_name)
    copy(image_path, new_path)

mk_file(path_set514_LRbicx4_prefix)
all_X4_img = os.listdir(path_valid_LR_bicubic_X4)
for png_img in mmengine.track_iter_progress(all_X4_img):
    image_path = osp.join(path_valid_LR_bicubic_X4, png_img)
    new_name = png_img[:-6] + png_img[-4:]
    new_path = osp.join(path_set514_LRbicx4_prefix, new_name)
    copy(image_path, new_path)
