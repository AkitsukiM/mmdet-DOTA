# 实验记录

## mmdetection安装：

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install mmcv-full
pip install -r requirements/build.txt
pip install -v -e .
```

## DOTA_devkit安装：

```shell
sudo apt install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
pip install shapely tqdm
```

## DOTA数据集创建

使用官网下载的数据集解压创建

```shell
# cd ~/Workspace/DOTA_devkit-master/
# mkdir /home/ubuntu/Dataset/DOTA/
# ln -s /home/ubuntu/Dataset/DOTA/ /home/marina/Workspace/Dataset/
python md5_calc.py --path /home/marina/Workspace/Dataset/DOTA/train.tar.gz
# cfb5007ada913241e02c24484e12d5d2
python md5_calc.py --path /home/marina/Workspace/Dataset/DOTA/val.tar.gz
# a53e74b0d69dacf3ffcb438accd60c45
tar -xzf /home/marina/Workspace/Dataset/DOTA/train.tar.gz -C /home/marina/Workspace/Dataset/DOTA/
tar -xzf /home/marina/Workspace/Dataset/DOTA/val.tar.gz -C /home/marina/Workspace/Dataset/DOTA/
python dir_list.py --path /home/marina/Workspace/Dataset/DOTA/train/images/ --output /home/marina/Workspace/Dataset/DOTA/train/trainset.txt
# 1411
python dir_list.py --path /home/marina/Workspace/Dataset/DOTA/val/images/ --output /home/marina/Workspace/Dataset/DOTA/val/valset.txt
# 458
```

## 一些修改

### 修改ImgSplit.py的main函数

以及其他一些修改

```shell
cd ~/Workspace/Dataset/
rm -rf DOTA-ImgSplit-COCO/
rm -rf DOTA-ImgSplit/
cd ../DOTA_devkit-master/
python ImgSplit.py
# 100%|███████████████████████████████████████████████████████████████████████████████████████| 1411/1411 [18:20<00:00,  1.28it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████| 458/458 [06:05<00:00,  1.25it/s]
```

### 修改DOTA2COCO.py为DOTA2COCO_rotated.py

以及其他一些修改

```shell
python DOTA2COCO_rotated.py
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 15749/15749 [06:14<00:00, 42.05it/s]
# 100%|███████████████████████████████████████████████████████████████████████████████████████| 5297/5297 [02:04<00:00, 42.70it/s]
python DOTA2COCO_rotated.py
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 60637/60637 [45:14<00:00, 22.34it/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 20579/20579 [16:37<00:00, 20.62it/s]
```

### 创建数据集软链接

```shell
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit/train/images /home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/train2017
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit/val/images /home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/val2017
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit/ /home/ubuntu/Dataset/
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/ /home/ubuntu/Dataset/
```

至此完成数据集路径搭建：

.../  
├─ Workspace/Dataset  
········├─ DOTA (link)  
················├─ train  
························├─ images  
································├─ P0000.png  
································├─ P0001.png  
································└─ ...  
························├─ labelTxt  
································├─ P0000.txt  
································├─ P0001.txt  
································└─ ...  
························└─ trainset.txt  
················└─ val  
························├─ images  
································├─ P0003.png  
································├─ P0004.png  
································└─ ...  
························├─ labelTxt  
································├─ P0003.txt  
································├─ P0004.txt  
································└─ ...  
························└─ valset.txt  
········├─ DOTA-ImgSplit  
················├─ train  
························├─ images  
································├─ P0000\_\_1\_\_0\_\_\_0.png  
································├─ P0000\_\_1\_\_0\_\_\_1648.png  
································└─ ...  
························└─ labelTxt  
································├─ P0000\_\_1\_\_0\_\_\_0.txt  
································├─ P0000\_\_1\_\_0\_\_\_1648.txt  
································└─ ...  
················└─ val  
························├─ images  
································├─ P0003\_\_1\_\_0\_\_\_0.png  
································├─ P0003\_\_1\_\_123\_\_\_0.png  
································└─ ...  
························└─ labelTxt  
································├─ P0003\_\_1\_\_0\_\_\_0.txt  
································├─ P0003\_\_1\_\_123\_\_\_0.txt  
································└─ ...  
········├─ DOTA-ImgSplit-COCO  
················├─ train2017 (link)  
················├─ val2017 (link)  
················└─ annotations  
························├─ instances_train2017.json  
························└─ instances_val2017.json  
········└─ ...  

### 修改mmdet config

修改mmdet config文件中的数据集类型为'Dota2CocoDataset'

修改路径、batchsize等

### 修改mmdet mmdet datasets

增加'Dota2CocoDataset'类，以及修改对应的__init__文件

从DOTA2COCO.py中复制类别信息

```
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
```

### 训练与测试

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/swin/cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-1x-cos_mssplit_coco_dota.py 8 > nohup.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/swin/swin-b_ms-test_max-1000.py ./work_dirs/save_1/epoch_12.pth 8 --out /home/marina/Workspace/DOTA_devkit-master/save_1_12.pkl --eval bbox
```

### 合并

修改dota_evaluation_task2.py

编写试作型HBB_evaluator.py

```shell
python HBB_evaluator.py --pkl save_1_12.pkl
```

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/03/30

