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
## POLYGON ANNOTATIONS 
path_file_annotations_POIs_polygons = '/root/autodl-tmp/SEM_rawdata/nature_scidata_heerlen_aachen_steel_annotations_polygon.csv'

# 1. 对数据集进行划分，并保证随机性可复现
random.seed(0)
split_rate = 0.2  # train:val=8:2

train_root = "./train"
mk_file(train_root)
val_root = "./val"
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

# 2. 两位专家及以上都同意的标注信息
dfAnnotationsPOIsAndPolygons = pd.read_csv(path_file_annotations_POIs_polygons, sep=";", converters={"point": ast.literal_eval, "polygon": ast.literal_eval})


images_all = {}

# 标注信息的整合
for idx_train in mmengine.track_iter_progress(range(len(dfAnnotationsPOIsAndPolygons['image_url']))):
    image_url = dfAnnotationsPOIsAndPolygons['image_url'][idx_train]
    # TODO 检查当前url在训练集中还是测试集中， 分别划分
    point = dfAnnotationsPOIsAndPolygons['point'][idx_train]
    polygon = dfAnnotationsPOIsAndPolygons['polygon'][idx_train]
    if image_url in images_all:
        images_all[image_url]['polygons'].append(polygon)
        images_all[image_url]['points'].append(point)
    else:
        images_all[image_url] = {}
        images_all[image_url]['polygons'] = [polygon]
        images_all[image_url]['points'] = [point]

# 3. 对训练和验证的image_train_all和image_val_all分别生成不同的MA_train_coco.json 和MA_val_coco.json
annotations_train = []
obj_count_train = 0
images_train = []
idx_train = 0

annotations_val = []
obj_count_val = 0
images_val = []
idx_val = 0

for key, val in mmengine.track_iter_progress(images_all.items()):
    filename = key
    if filename in eval_index:
        img_path = osp.join(path_folder_images_png, filename)
        height, width = mmcv.imread(img_path).shape[:2]
        images_val.append(dict(
            id=idx_val,
            file_name=filename,
            height=height,
            width=width
        ))

        bboxes = []
        labels = []
        masks = []
        for polygon in val['polygons']:
            polygon = np.array(polygon)
            x_min, y_min, x_max, y_max = (
                min(polygon[:, 0]), min(polygon[:, 1]), max(polygon[:, 0]), max(polygon[:, 1])
            )
            data_anno = dict(
                image_id=idx_val,
                id=obj_count_val,
                category_id=0,
                bbox=[x_min, y_min, x_max, y_max],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[polygon],
                iscrowd=0
            )
            annotations_val.append(data_anno)
            obj_count_val += 1
        
        idx_val += 1

    else:
        img_path = osp.join(path_folder_images_png, filename)
        height, width = mmcv.imread(img_path).shape[:2]
        images_train.append(dict(
            id=idx_train,
            file_name=filename,
            height=height,
            width=width
        ))

        bboxes = []
        labels = []
        masks = []
        for polygon in val['polygons']:
            polygon = np.array(polygon)
            x_min, y_min, x_max, y_max = (
                min(polygon[:, 0]), min(polygon[:, 1]), max(polygon[:, 0]), max(polygon[:, 1])
            )
            polygon = [p for x in polygon for p in x]
            data_anno = dict(
                image_id=idx_train,
                id=obj_count_train,
                category_id=0,
                bbox=[x_min, y_min, x_max, y_max],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[polygon],
                iscrowd=0
            )
            annotations_train.append(data_anno)
            obj_count_train += 1
        
        idx_train += 1

MA_train_coco_json = dict(
    images=images_train,
    annotations=annotations_train,
    categories=[{'id':0, 'name': 'MA'}]
)
MA_val_coco_json = dict(
    images=images_val,
    annotations=annotations_val,
    categories=[{'id':0, 'name': 'MA'}]
)

mmengine.dump(MA_train_coco_json, './MA_train_coco.json') 
mmengine.dump(MA_val_coco_json, './MA_val_coco.json')





