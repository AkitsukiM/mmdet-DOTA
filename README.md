# DOTA数据集目标检测相关工作

## 目前进展

1.  DOTA数据集向COCO数据集格式的转换，包括图片分割、json文件生成、MMDet中对应的数据集类型搭建

2.  DOTA数据集在MMDet提供的cascade-rcnn框架下的检测

3.  MMDet测试得到的pkl结果文件的读取、合并、mAP计算与可视化

> 训练时model.test_cfg.rcnn.max_per_img=100，测试时model.test_cfg.rcnn.max_per_img=1000

4.  重新计算了DOTA数据集的均值与方差

5.  将cascade-rcnn框架的backbone由resnet50替换为swin-t

6.  由1x加到3x（即12epochs加到36epochs）

7.  引入了ms_crop（多尺度随机裁剪）和fp16

8.  参考ATSS引入了多尺度测试

9.  引入了水平翻转与垂直翻转各占一定比例的数据增强策略

10. 参考s2anet引入了旋转数据增强，并配套进行了数据集加载模块的改写以及可视化

11. 将swin-t升级为swin-b，需对应修改batch_size与init_lr

12. 引入了学习率余弦退火策略

13. 参考s2anet引入了训练集多尺度分割

> 测试集多尺度分割反而导致性能下降

以及其他一些小工具

## 如何使用

复制粘贴替换即可，具体操作请参考md文件

## 当前最高性能（step 1-12）

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.482
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.779
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.540
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.507
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.690
```

after merge:

```
map: 0.7830074123341149
classaps:  [96.94026914 86.73521017 46.66968419 68.36142512 72.35036478 84.95429386
 93.71276841 95.66885508 71.70114309 85.77073779 67.09983308 71.35375173
 86.16385033 71.58292262 75.4460091 ]
```

## 参考资料

DOTA数据集以及相关工具简介：https://zhuanlan.zhihu.com/p/355862906

DOTA数据集网站：https://captain-whu.github.io/DOTA/dataset.html

DOTA devkit：https://github.com/CAPTAIN-WHU/DOTA_devkit

mmdetection：https://github.com/open-mmlab/mmdetection

Swin Transformer：https://github.com/SwinTransformer/Swin-Transformer-Object-Detection

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/03/26

