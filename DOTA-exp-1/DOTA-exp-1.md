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
```

## 一些修改

### 修改ImgSplit.py的main函数

```
# example usage of ImgSplit
split_train = splitbase(r'/home/marina/Workspace/Dataset/DOTA/train/', r'/home/marina/Workspace/Dataset/DOTA-ImgSplit/train/', gap=200)
split_train.splitdata(1)
split_val = splitbase(r'/home/marina/Workspace/Dataset/DOTA/val/', r'/home/marina/Workspace/Dataset/DOTA-ImgSplit/val/', gap=200)
split_val.splitdata(1)
```

以及其他一些修改

```shell
python ImgSplit.py
```

### 修改DOTA2COCO.py的main函数

```
DOTA2COCO(r'/home/marina/Workspace/Dataset/DOTA-ImgSplit/train/', r'/home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/annotations/instances_train2017.json')
DOTA2COCO(r'/home/marina/Workspace/Dataset/DOTA-ImgSplit/val/', r'/home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/annotations/instances_val2017.json')
```

以及其他一些修改

```shell
python DOTA2COCO.py
```

### 创建数据集软链接

```shell
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit/train/images /home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/train2017
ln -s /home/marina/Workspace/Dataset/DOTA-ImgSplit/val/images /home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/val2017
```

至此完成数据集路径搭建：

.../  
├─ Dataset  
········├─ DOTA  
················├─ train  
························├─ images  
································├─ P0000.png  
································├─ P0001.png  
································└─ ...  
························└─ labelTxt  
································├─ P0000.txt  
································├─ P0001.txt  
································└─ ...  
················├─ val  
························├─ images  
································├─ P0003.png  
································├─ P0004.png  
································└─ ...  
························├─ labelTxt  
································├─ P0003.txt  
································├─ P0004.txt  
································└─ ...  
························└─ valset.txt  
················├─ trainval  
················└─ test  
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

路径为'/home/marina/Workspace/Dataset/DOTA-ImgSplit-COCO/'

修改batchsize

### 修改mmdet mmdet datasets

增加'Dota2CocoDataset'类，以及修改对应的__init__文件

从DOTA2COCO.py中复制类别信息

```
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
```

### 训练

```shell
python ./tools/train.py ./configs/paa/paa_r50_fpn_1x_coco_dota_test_1.py
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/dist_train.sh ./configs/paa/paa_r50_fpn_1x_coco_dota_test_1.py 4
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup ./tools/dist_train.sh ./configs/paa/paa_r50_fpn_1x_coco_dota_test_1.py 4 > nohup.log 2>&1 &
```

### 测试

```shell
python ./tools/test.py ./configs/paa/paa_r50_fpn_1x_coco_dota_test_1.py ./work_dirs/paa_r50_fpn_1x_coco_dota_test_1/epoch_12.pth --out /home/marina/Workspace/DOTA_devkit-master/xxx.pkl --eval bbox
```

得到pkl文件

### 合并

修改dota_evaluation_task2.py中的parse_gt函数

编写read_pkl_task2.py并运行，包含可视化

-----

Copyright (c) 2022 Marina Akitsuki. All rights reserved.

Date modified: 2022/01/10

