import random
import os
import numpy as np
try:
    import torch
except:
    pass
import cv2

def seed_everything(seed):
    "seed値を一括指定"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path):
    """
    pathからimageの配列を得る
    """
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        print(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def calc_black_whiteArea(bw_image):
    image_size = bw_image.size
    whitePixels = cv2.countNonZero(bw_image)
    whiteAreaRatio = (whitePixels/image_size)

    return whiteAreaRatio
