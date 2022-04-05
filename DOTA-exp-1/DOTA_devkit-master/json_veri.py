import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

class Json_veri(object):
    """
    """
    def __init__(self, jsonpath, wmax = 1024, hmax = 1024, max_per_img = 675):
        # ref: https://cocodataset.org/#format-data
        # ref: https://github.com/cocodataset/cocoapi

        self.jsonpath = jsonpath
        self.cocoGt = COCO(self.jsonpath)

        self.imgIds = self.cocoGt.getImgIds()
        self.imgs   = self.cocoGt.loadImgs(self.imgIds)
        self.catIds = self.cocoGt.getCatIds()
        self.cats   = self.cocoGt.loadCats(self.catIds)
        self.gtIds  = self.cocoGt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds)
        self.gts    = self.cocoGt.loadAnns(self.gtIds)

        self.wmax = wmax
        self.hmax = hmax
        self.max_per_img = max_per_img

    def check_imgsize(self):
        for img in tqdm(self.imgs):
            if img["width"] > self.wmax or img["height"] > self.hmax:
                print(img["file_name"] + ":", "w-" + img["width"], "h-" + img["height"])

    def check_bboxsize(self):
        dict = {}
        img_with_0_bboxes = 0
        img_with_too_many_bboxes = 0
        max_num = 0
        for imgId in self.imgIds:
            dict[imgId] = []
        for gt in self.gts:
            dict[gt["image_id"]].append(gt["bbox"])
        for value in tqdm(dict.values()):
            if len(value) == 0:                 img_with_0_bboxes += 1
            if len(value) > self.max_per_img:   img_with_too_many_bboxes += 1
            if len(value) > max_num:            max_num = len(value)
        print("img_with_0_bboxes:",             img_with_0_bboxes)
        print("img_with_too_many_bboxes:",      img_with_too_many_bboxes)
        print("max_num:",                       max_num)

    def main(self):
        self.check_imgsize()
        self.check_bboxsize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonpath", type = str, default = "./instances_val2017.json", help = "jsonpath")
    opt = parser.parse_args()

    Json_veri(jsonpath = opt.jsonpath).main()

