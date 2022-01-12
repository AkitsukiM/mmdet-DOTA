# 实验记录

## 编写config

基于mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py与cascade_rcnn_r50_fpn.py更改

## 训练

```shell
conda activate openmmlab
# original
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_original_fpn_1x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_original_fpn_1x_coco_dota.py 4 > nohup.log 2>&1 &
# swin-t-p4-w7
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4 > nohup.log 2>&1 &
```

## 测试original

python ./tools/test.py ./configs/swin/cascade_rcnn_original_fpn_1x_coco_dota.py ./work_dirs/cascade_rcnn_original_fpn_1x_coco_dota/epoch_12.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_original_fpn_1x_coco_dota.pkl --eval bbox

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.628
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.445
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.277
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.639
```

after merge:

```
map: 0.6583400669209473
classaps:  [90.76794053 72.89804461 45.54077481 61.98607558 50.06718993 65.37453132
 63.4426761  94.4160992  57.14288873 78.68369016 58.68007134 65.60383328
 73.8996194  56.7460234  52.26064198]
```

## 测试swin-t-p4-w7

python ./tools/test.py ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py ./work_dirs/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota/epoch_12.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.pkl --eval bbox

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.671
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.656
```

after merge:

```
map: 0.6999776064171848
classaps:  [93.23392502 79.45959511 46.82152478 65.5987476  52.03374648 66.93252033
 65.0563629  95.17581591 56.73506884 80.93194335 68.94698407 72.71591187
 74.7365891  63.93881485 67.64885941]
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/12

