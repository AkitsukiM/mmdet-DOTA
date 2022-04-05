import dota_utils as util
import os
import cv2
import json

from tqdm import tqdm
import numpy as np


wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']


# from https://github.com/csuhan/s2anet/blob/master/mmdet/core/bbox/transforms_rotated.py ##### #####
# np.float -> np.float32

def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]

def poly_to_rotated_box_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_box:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float32(pt2[1] - pt1[1]), np.float32(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float32(pt4[1] - pt1[1]), np.float32(pt4[0] - pt1[0]))

    angle = norm_angle(angle)

    x_ctr = np.float32(pt1[0] + pt3[0]) / 2
    y_ctr = np.float32(pt1[1] + pt3[1]) / 2
    rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
    return rotated_box


def DOTA2COCO(srcpath, destfile, is_rotated = False, max_per_img = 0):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    # remakedirs
    destpath = os.path.abspath(os.path.join(destfile, ".."))
    # print(destpath)
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    if os.path.exists(destfile):
        os.remove(destfile)

    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2018',
           'description': 'This is 1.0 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '1.0',
           'year': 2018}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(wordname_15):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in tqdm(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            if max_per_img != 0 and len(objects) > max_per_img:
                image_id = image_id + 1
                continue
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = wordname_15.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0

                if is_rotated:
                    x_ctr, y_ctr, width, height, angle = poly_to_rotated_box_single(obj['poly'][0:8])
                    single_obj['bbox'] = x_ctr, y_ctr, width, height, angle
                else:
                    xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                             max(obj['poly'][0::2]), max(obj['poly'][1::2])
                    width, height = xmax - xmin, ymax - ymin
                    single_obj['bbox'] = xmin, ymin, width, height

                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    dataset_root = "/home/marina/Workspace/Dataset/"

    srcpath_train           = dataset_root + r'DOTA-ImgSplit/train/'
    destfile_train          = dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_train2017.json'
    destfile_train_mssplit  = dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_train2017_mssplit.json'
    srcpath_val             = dataset_root + r'DOTA-ImgSplit/val/'
    destfile_val            = dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_val2017.json'
    destfile_val_mssplit    = dataset_root + r'DOTA-ImgSplit-COCO/annotations/instances_val2017_mssplit.json'

    # DOTA2COCO(srcpath_train, destfile_train, is_rotated = True)
    DOTA2COCO(srcpath_train, destfile_train_mssplit, is_rotated = True, max_per_img = 675)

    # DOTA2COCO(srcpath_val, destfile_val)
    # DOTA2COCO(srcpath_val, destfile_val_mssplit)

