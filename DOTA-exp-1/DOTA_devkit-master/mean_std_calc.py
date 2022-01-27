import os
import cv2
import argparse
import math
import numpy as np
from tqdm import tqdm


def mean_std_calc(dir):
    imgname_list = os.listdir(dir)
    # print(imgname_list)
    b_sum = 0.0
    g_sum = 0.0
    r_sum = 0.0
    b2_sum = 0.0
    g2_sum = 0.0
    r2_sum = 0.0
    pixel_num = 0

    print("Now calculating mean:")
    for imgname in tqdm(imgname_list):
        img_path = os.path.join(dir, imgname)
        img = cv2.imread(img_path)
        if not isinstance(img, np.ndarray):
            print(img_path)
            continue
        h, w, c = img.shape
        # print(h, w, c)
        b, g, r = np.sum(img, axis=(0, 1))
        # print(b, g, r)
        b_sum += b
        g_sum += g
        r_sum += r
        pixel_num += h * w
        # break

    b_mean = b_sum / pixel_num
    g_mean = g_sum / pixel_num
    r_mean = r_sum / pixel_num
    print("b_mean, g_mean, r_mean:", b_mean, g_mean, r_mean)
    bgr_mean = np.asarray([b_mean, g_mean, r_mean])

    print("Now calculating std:")
    for imgname in tqdm(imgname_list):
        img_path = os.path.join(dir, imgname)
        img = cv2.imread(img_path)
        if not isinstance(img, np.ndarray):
            # print(img_path)
            continue
        # h, w, c = img.shape
        # print(h, w, c)
        img = img - bgr_mean
        img2 = img ** 2
        b2, g2, r2 = np.sum(img2, axis=(0, 1))
        # print(b2, g2, r2)
        b2_sum += b2
        g2_sum += g2
        r2_sum += r2
        # pixel_num += h * w
        # break

    b2_mean = b2_sum / pixel_num
    g2_mean = g2_sum / pixel_num
    r2_mean = r2_sum / pixel_num
    print("b_std, g_std, r_std:", math.sqrt(b2_mean), math.sqrt(g2_mean), math.sqrt(r2_mean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type = str, default = None, help = "")
    opt = parser.parse_args()

    mean_std_calc(opt.dir)

