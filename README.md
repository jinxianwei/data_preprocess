# data_preprocess
## 通用工具
- split_train_val.py 将数据分割为训练集和验证集
- move_rename.py 复制并改变文件名称
- mmdet_browse_data.py mmdetection中检查data preprocesser后的数据显示
- mmseg_browse_data.py mmsegmentation中检查data preprocesser后的数据显示
- hr2lr_data_convert.py 构造DIV2K数据格式（高分辨率图像bicubic下采样获得x2、x3、x4的分辨率）

## 特殊数据集工具
- steeldata_analyse.py 海伦凯勒数据集拓扑标注转化为coco数据集格式（目标检测和实例分割）
- TODO 海伦凯勒数据集转化为VOC语义分割数据集