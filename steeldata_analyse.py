import os.path as osp
import mmcv
from tqdm import tqdm

import pandas as pd 
import ast
import mmengine
import numpy as np



## IMAGE FILES
path_folder_images_png = '/root/autodl-tmp/SEM_rawdata/PNG/'
path_folder_images_tiff = '/root/autodl-tmp/SEM_rawdata/TIFF/'

## STEEL SAMPLE METADATA
path_file_metadata = "/root/autodl-tmp/SEM_rawdata/nature_scidata_steel_metadata.csv"

## POI ANNOTATIONS
path_file_annotations_POIcsv = "/root/autodl-tmp/SEM_rawdata/nature_scidata_heerlen_aachen_steel_annotations_POI.csv"

## POLYGON ANNOTATIONS 
path_file_annotations_POIs_polygons_pickle = '/root/autodl-tmp/SEM_rawdata/nature_scidata_heerlen_aachen_steel_annotations_polygon.pickle'
path_file_annotations_POIs_polygons = '/root/autodl-tmp/SEM_rawdata/nature_scidata_heerlen_aachen_steel_annotations_polygon.csv'

## MORPHOLOGICAL FEATURES
path_file_annotations_morphology = '/root/autodl-tmp/SEM_rawdata/nature_scidata_heerlen_aachen_steel_morph.pickle'

## CALCULATED CONTOURS
path_file_AnnotationsPOIsAndPolygonsShapely = '/root/autodl-tmp/SEM_rawdata/nature_scidata_dfPOIPolygonContourShapely.pickle'
path_file_Evaluations = '/root/autodl-tmp/SEM_rawdata/nature_scidata_dfEvaluation.pickle'


# dfAnnotationsPOI = pd.read_csv(path_file_annotations_POIcsv, sep=";")
# print("Number of images: ", len(dfAnnotationsPOI))
# dfAnnotationsPOI.head()

# TODO 需要先将图像数据集进行划分

# 两位专家及以上都同意的标注信息
dfAnnotationsPOIsAndPolygons = pd.read_csv(path_file_annotations_POIs_polygons, sep=";", converters={"point": ast.literal_eval, "polygon": ast.literal_eval})


images_all = {}


for idx in mmengine.track_iter_progress(range(len(dfAnnotationsPOIsAndPolygons['image_url']))):
    image_url = dfAnnotationsPOIsAndPolygons['image_url'][idx]
    # TODO 检查当前url在训练集中还是测试集中， 分别划分
    point = dfAnnotationsPOIsAndPolygons['point'][idx]
    polygon = dfAnnotationsPOIsAndPolygons['polygon'][idx]
    if image_url in images_all:
        images_all[image_url]['polygons'].append(polygon)
        images_all[image_url]['points'].append(point)
    else:
        images_all[image_url] = {}
        images_all[image_url]['polygons'] = [polygon]
        images_all[image_url]['points'] = [point]

# TODO 对训练和验证的image_train_all和image_val_all分别生成不同的MA_train_coco.json 和MA_val_coco.json

# print(len(images_all))

annotations = []
obj_count = 0
images = []
idx = 0

for key, val in mmengine.track_iter_progress(images_all.items()):
    filename = key
    img_path = osp.join(path_folder_images_png, filename)
    height, width = mmcv.imread(img_path).shape[:2]
    images.append(dict(
        id=idx,
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
            image_id=idx,
            id=obj_count,
            category_id=0,
            bbox=[x_min, y_min, x_max, y_max],
            area=(x_max - x_min) * (y_max - y_min),
            segmentation=[polygon],
            iscrowd=0
        )
        annotations.append(data_anno)
        obj_count += 1
    
    idx += 1

coco_format_json = dict(
    images=images,
    annotations=annotations,
    categories=[{'id':0, 'name': 'MA'}]
)

mmengine.dump(coco_format_json, './coco_format.json') 





