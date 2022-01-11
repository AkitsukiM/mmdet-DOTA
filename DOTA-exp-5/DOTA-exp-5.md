# 实验记录

## 编写config

基于mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py与cascade_rcnn_r50_fpn.py更改

## 训练

```shell
conda activate openmmlab
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco_dota.py 4 > nohup.log 2>&1 &
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/11

