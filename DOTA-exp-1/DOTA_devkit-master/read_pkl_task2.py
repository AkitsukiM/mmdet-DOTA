# By using the command
# >>> python ./tools/test.py ./configs/***.py work_dirs/***/latest.pth --out ***.pkl --eval bbox
# we will get a file named ***.pkl
# 
# Now we should read this file and change the result into the format introduced in
# https://captain-whu.github.io/DOTA/tasks.html
# so that we can use DOTA_devkit/ResultMerge.py to merge the result
# 

from pycocotools.coco import COCO
import os
import numpy as np
import pylab
import pickle
from ResultMerge import mergebyrec, mergebypoly
from dota_evaluation_task2 import voc_eval

import cv2
import shutil
from tqdm import tqdm
from visualize import *

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# 工作路径
output_root = "/home/marina/Workspace/DOTA_devkit-master/"
# 工作路径下的子路径
saveDir = output_root + "Task2/"
mergeDir = output_root + "Task2_merge/"
visualDir = output_root + "visual/"
resultPickle = output_root + "xxx.pkl"

# remakedirs
if os.path.exists(saveDir):
    shutil.rmtree(saveDir)
if os.path.exists(mergeDir):
    shutil.rmtree(mergeDir)
if os.path.exists(visualDir):
    shutil.rmtree(visualDir)

os.makedirs(saveDir)
os.makedirs(mergeDir)
os.makedirs(visualDir)

# merge前的所有类别的txt文件
split_detpath = os.path.join(saveDir, 'Task2_{:s}.txt')
# merge后的所有类别的txt文件
detpath = os.path.join(mergeDir, 'Task2_{:s}.txt')

# 数据集根路径
dataset_root = "/home/marina/Workspace/Dataset/"
# 验证集的原始标注txt文件
annopath = dataset_root + r'DOTA/val/labelTxt/{:s}.txt'
# 验证集的原始图片名文件
imagesetfile = dataset_root + r'DOTA/val/valset.txt'
# 分割前的图片路径
ori_img_path = dataset_root + r'DOTA/val/images/'
# 分割后的图片路径
split_img_path = dataset_root + r'DOTA-ImgSplit/val/images/'
# 分割后的json文件路径
split_json_path = dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_val2017.json'
# 分割后的标注txt文件
split_annopath = dataset_root + r'DOTA-ImgSplit/val/labelTxt/{:s}.txt'
# 分割后的图片名文件
split_imagesetfile = dataset_root + r'DOTA-ImgSplit/val/valset.txt'

classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

val_coco = COCO(split_json_path) # 读取json文件
boxes_result = pickle.load(open(resultPickle, 'rb')) # 读取pkl文件

cats = val_coco.loadCats(val_coco.getCatIds())
categories = [cat["name"] for cat in cats]
ids = [cat["id"] for cat in cats]
ind_to_label_map = {ind: label for ind, label in zip(ids, categories)}

# 至此，ind_to_label_map的结果是
# {1: "plane", 2: "baseball-diamond", ..., 15: "helicopter"}

imgs = val_coco.loadImgs(val_coco.getImgIds())
images = [img["file_name"] for img in imgs]
img_ids = [img["id"] for img in imgs]
ind_to_imagefile_map = {ind: file_name for ind, file_name in zip(img_ids, images)}

# 至此，ind_to_imagefile_map的结果是
# {1: "P1156__1__1648___824.png", 2: "P1513__1__0___2472.png", ..., 5297: "P1699__1__1648___2472.png"}

# 把ind_to_imagefile_map的value写到split_imagesetfile里去
with open(split_imagesetfile, 'w') as f:
    for file_name in images:
        f.write(file_name[:-4] + '\n')

# 把未merge的结果写到saveDir下
for cat_id in ind_to_label_map.keys():
    with open(os.path.join(saveDir, "Task2_" + ind_to_label_map[cat_id] + ".txt"), 'w') as f:
        for image_id, result_per_image in enumerate(boxes_result):
            cat_bboxes = result_per_image[cat_id - 1]
            for det in cat_bboxes:
                bbox = det[-5:-1]
                score = str(round(det[-1], 5))
                imgname = ind_to_imagefile_map[image_id + 1][:-4] # 去掉拓展名

                xmin_str = str(int(bbox[0]))
                ymin_str = str(int(bbox[1]))
                xmax_str = str(int(bbox[2]))
                ymax_str = str(int(bbox[3]))

                f.write(imgname + ' ' + score + ' ' + 
                        xmin_str + ' ' + ymin_str + ' ' +
                        xmax_str + ' ' + ymax_str + ' ' + '\n')

# saveDir的可视化
print("\nNOW VISUALIZING SAVE_DIR...")
ind = 1
for result_per_image in tqdm(boxes_result):
    img_path = os.path.join(split_img_path, ind_to_imagefile_map[ind])
    img = cv2.imread(img_path)
    bboxes_prd = np.zeros(6)
    bboxes_prd = np.expand_dims(bboxes_prd, axis = 0)
    for cat_id in ind_to_label_map.keys():
        cat_bboxes = result_per_image[cat_id - 1]
        for det in cat_bboxes:
            bboxes_new = np.append(det[-5:], cat_id - 1)
            bboxes_new = np.expand_dims(bboxes_new, axis = 0)
            bboxes_prd = np.concatenate((bboxes_prd, bboxes_new), axis = 0)

    boxes = bboxes_prd[1:, :4]
    class_inds = bboxes_prd[1:, 5].astype(np.int32)
    scores = bboxes_prd[1:, 4]

    visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = classnames)
    path = os.path.join(visualDir, ind_to_imagefile_map[ind])
    cv2.imwrite(path, img)
    ind += 1

# 利用dota工具包进行merge，将每一类在split上的检测结果合并到大图上
# merge工具自带偏移量计算
# task1: mergebypoly
# task2: mergebyrec
print("\nNOW MERGING...")
mergebyrec(saveDir, mergeDir)

# mergeDir的可视化
print("\nNOW VISUALIZING MERGE_DIR...")
imgname_xyxysc_dict = {}

for cat_id in ind_to_label_map.keys():
    with open(os.path.join(mergeDir, "Task2_" + ind_to_label_map[cat_id] + ".txt"), 'r') as f:
        print("Working with Task2_" + ind_to_label_map[cat_id] + ".txt")
        lines = f.readlines()
        splitlines = [x.strip().split(' ')  for x in lines]
        for splitline in tqdm(splitlines):
            if (len(splitline) != 6):
                continue
            imgname = splitline[0]
            bboxes_new = np.asarray([float(splitline[2]),
                                     float(splitline[3]),
                                     float(splitline[4]),
                                     float(splitline[5]),
                                     float(splitline[1]),
                                     float(cat_id - 1)])
            bboxes_new = np.expand_dims(bboxes_new, axis = 0)
            if (imgname not in imgname_xyxysc_dict):
                imgname_xyxysc_dict[imgname] = bboxes_new
            else:
                bboxes_prd = imgname_xyxysc_dict[imgname]
                imgname_xyxysc_dict[imgname] = np.concatenate((bboxes_prd, bboxes_new), axis = 0)

for imgname in tqdm(imgname_xyxysc_dict.keys()):
    img_path = os.path.join(ori_img_path, imgname + ".png")
    img = cv2.imread(img_path)
    bboxes_prd = imgname_xyxysc_dict[imgname]
    boxes = bboxes_prd[..., :4]
    class_inds = bboxes_prd[..., 5].astype(np.int32)
    scores = bboxes_prd[..., 4]

    visualize_boxes(image = img, boxes = boxes, labels = class_inds, probs = scores, class_labels = classnames)
    path = os.path.join(visualDir, imgname + ".png")
    cv2.imwrite(path, img)

# 进行voc评测
# 照抄dota_evaluation_task2的main函数即可
print("\nNOW EVALUATING AP50...")
classaps = []
map = 0
for classname in classnames:
    print('classname:', classname)
    rec, prec, ap = voc_eval(detpath,
            annopath,
            imagesetfile,
            classname,
            ovthresh=0.5,
            use_07_metric=False)
    map = map + ap
    #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
    print('ap: ', ap)
    classaps.append(ap)

    ## uncomment to plot p-r curve for each category
    # plt.figure(figsize=(8,4))
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.plot(rec, prec)
    # plt.show()
map = map/len(classnames)
print('map:', map)
classaps = 100*np.array(classaps)
print('classaps: ', classaps)

