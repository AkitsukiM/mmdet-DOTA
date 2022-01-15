# 实验记录

## 移植PPH

特别注意config的更改，以及移除

```
from timm.models.layers import trunc_normal_, DropPath
```

## 训练

```shell
conda activate openmmlab
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/pph/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco_dota.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ./tools/dist_train.sh ./configs/pph/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco_dota.py 4 > nohup.log 2>&1 &
```

## 测试

```shell
python ./tools/test.py ./configs/pph/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco_dota.py ./work_dirs/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco_dota/epoch_36.pth --out /home/marina/Workspace/DOTA_devkit-master/pph_conformer_small_p32_fpn_mstrain_crop_100tokens_3x_coco_dota.pkl --eval bbox
```

before merge:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.526
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.642
```

after merge:

```
map: 0.6074796200568727
classaps:  [85.88505428 65.95368292 40.94427667 56.49227469 46.0720221  58.75908671
 59.06121431 91.24014236 53.49790052 73.35641749 47.28463055 60.17724579
 63.26320069 65.78612354 43.44615747]
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/12

