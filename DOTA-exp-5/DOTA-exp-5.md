# 实验记录

## 编写config

cascade_rcnn_r50_fpn_1x_coco_dota.py: from '../_base_/models/cascade_rcnn_r50_fpn.py'

cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py: based on './cascade_rcnn_r50_fpn_1x_coco_dota.py'; from './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py' and './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

## 训练与测试

```shell
conda activate openmmlab

# swin-t-p4-w7_fpn_fp16_ms-crop-3x
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4 > nohup.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python ./tools/test.py ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py ./work_dirs/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota/epoch_36.pth --out /home/marina/Workspace/DOTA_devkit-master/a4.pkl --eval bbox
python HBB_pkl_reader.py --pkl_name a4.pkl
```

## 性能报告

### swin-t-p4-w7_fpn_fp16_ms-crop-3x

改进点：

1. 更改了test_cfg.rcnn.nms.iou_threshold和test_cfg.rcnn.max_per_img=1000

2. 更改了img_norm_cfg

3. 加入了垂直翻转的训练增强

4. 简单的多尺度测试

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.758
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.557
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.604
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.693
```

after merge:

```
map: 0.7556119451525671
classaps:  [95.46223742 83.28518442 46.00224732 69.46277421 67.64813776 84.13496228
 91.84358913 95.16287823 64.48185274 82.96858317 62.38077646 68.78760332
 84.05620859 70.7539107  66.98697197]
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/02/24

