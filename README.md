# DOTA数据集目标检测相关工作

## 目前进展

1.  DOTA数据集向COCO数据集格式的转换，包括图片分割、json文件生成、MMDet中对应的数据集类型搭建

2.  DOTA数据集在MMDet提供的cascade-rcnn框架下的检测

3.  MMDet测试得到的pkl结果文件的读取、合并、mAP计算与可视化

> 训练时model.test_cfg.rcnn.max_per_img=100，测试时model.test_cfg.rcnn.max_per_img=1000

4.  重新计算了DOTA数据集的均值与方差

5.  将cascade-rcnn框架的backbone由resnet50替换为swin-t

6.  将swin-t升级为swin-b，需对应修改batch_size与init_lr

7.  引入了ms_crop（多尺度随机裁剪）和fp16

8.  由1x加到3x（即12epochs加到36epochs）

> 实际swin-b 3x可能存在过拟合，性能不如1x

9.  引入了多尺度测试

> ./mmdet/core/post_processing/merge_augs.py使用了torch.stack().mean(dim=0)，原因不明

10. 引入了水平翻转与垂直翻转各占一定比例的数据增强策略；参考s2anet引入了旋转数据增强，并配套进行了数据集加载模块的改写以及可视化

11. 引入了学习率余弦退火策略

> 目前无法确定by_epoch取True还是False，虽然最高性能是取True但是定论证据不足

12. 参考s2anet引入了训练集多尺度分割

> 倍率0.5会造成GeForce RTX 2080Ti 11019MiB "cuda out of memory"，原因不明
> 因此我们目前使用倍率0.75/1.0/1.5，并在生成json文件时滤除gt数大于675的图（675是倍率1.0的最大单图gt数）
> 测试集多尺度分割反而导致性能下降

以及其他一些小工具

## 如何使用

复制粘贴替换即可，具体操作请参考md文件

## 当前最高性能

./configs/swin/cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-1x_cos-e_mssplit_coco_dota.py

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.503
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.798
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.528
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.643
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.703
```

after merge:

```
map: 0.8016679360096834
classaps:  [96.77909137 85.34528914 54.10785727 71.64477706 76.69326955 85.82405511
 94.22972774 95.54314411 68.39500636 87.4978248  65.48651439 76.18949556
 86.47673079 75.92745768 82.36166308]
```

## 参考资料

DOTA数据集以及相关工具简介：https://zhuanlan.zhihu.com/p/355862906

DOTA数据集网站：https://captain-whu.github.io/DOTA/dataset.html

DOTA devkit：https://github.com/CAPTAIN-WHU/DOTA_devkit

mmdetection：https://github.com/open-mmlab/mmdetection

Swin Transformer：https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/04/05

