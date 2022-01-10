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

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/10

