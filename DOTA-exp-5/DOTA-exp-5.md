# 实验记录

## 编写config

cascade_rcnn_r50_fpn_1x_coco_dota.py: from '../_base_/models/cascade_rcnn_r50_fpn.py'

cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py: based on './cascade_rcnn_r50_fpn_1x_coco_dota.py'; from './mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'

cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py: based on './cascade_rcnn_r50_fpn_1x_coco_dota.py'; from './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py' and './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py: based on './cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py'; from './mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'

## 训练与测试

```shell
conda activate openmmlab
# r50
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_r50_fpn_1x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_r50_fpn_1x_coco_dota.py 4 > nohup.log 2>&1 &
python ./tools/test.py ./configs/swin/cascade_rcnn_r50_fpn_1x_coco_dota.py ./work_dirs/cascade_rcnn_r50_fpn_1x_coco_dota/epoch_12.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_r50_fpn_1x_coco_dota.pkl --eval bbox
# swin-t-p4-w7
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4 > nohup.log 2>&1 &
python ./tools/test.py ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py ./work_dirs/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota/epoch_12.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.pkl --eval bbox
# swin-t-p4-w7_fpn_fp16_ms-crop-3x
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4 > nohup.log 2>&1 &
python ./tools/test.py ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py ./work_dirs/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota/epoch_36.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.pkl --eval bbox
# swin-s-p4-w7_fpn_fp16_ms-crop-3x
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 8 > nohup.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py 4 --auto-resume > nohup.log 2>&1
python ./tools/test.py ./configs/swin/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.py ./work_dirs/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota/epoch_36.pth --out /home/marina/Workspace/DOTA_devkit-master/cascade_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_dota.pkl --eval bbox
```

## 性能报告

### r50

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

### swin-t-p4-w7

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

### swin-t-p4-w7_fpn_fp16_ms-crop-3x

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.694
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.517
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.676
```

after merge:

```
map: 0.714853748364288
classaps:  [95.67122739 83.7797953  53.18352561 66.00858888 50.76746413 69.73810778
 64.58744955 95.59754055 63.51988706 79.12747267 60.43509387 70.71586954
 79.23891987 68.31138752 71.59829282]
```

### swin-s-p4-w7_fpn_fp16_ms-crop-3x

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.700
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.492
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.676
```

after merge:

```
map: 0.7234553964074001
classaps:  [95.44750255 85.095932   50.77639738 69.72405037 50.01056693 69.87040919
 64.67279377 95.146655   66.33602179 78.84093638 61.44221814 74.6535651
 79.87794457 66.30418112 76.9839203 ]
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/15

