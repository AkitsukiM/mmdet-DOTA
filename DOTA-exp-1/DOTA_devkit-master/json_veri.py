import argparse
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

class Json_veri(object):
    """
    """
    def __init__(self, jsonpath, wmax = 1024, hmax = 1024):
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

    def check_imgsize(self):
        for img in tqdm(self.imgs):
            if img["width"] > self.wmax or img["height"] > self.hmax:
                print(img["file_name"] + ":", "w-" + img["width"], "h-" + img["height"])

    def check_bboxsize(self):
        dict = {}
        empty_bbox = 0
        for imgId in self.imgIds:
            dict[imgId] = []
        for gt in self.gts:
            dict[gt["image_id"]].append(gt["bbox"])
        for value in tqdm(dict.values()):
            if len(value) == 0: empty_bbox += 1
        print("empty_bbox:", empty_bbox)

    def main(self):
        self.check_imgsize()
        self.check_bboxsize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonpath", type = str, default = "./instances_val2017.json", help = "jsonpath")
    opt = parser.parse_args()

    Json_veri(jsonpath = opt.jsonpath).main()

