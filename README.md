# data_preprocess

## 文件描述
处理扫描电镜原始标注数据，转化到mmlab支持的coco格式（data split 和 train val.json文件的生成）
使用mmdetection中的分析工具
tools/analysis_tools/browse_dataset.py 验证mask rcnn实例分割中经过data_preprocesser后图像的状态