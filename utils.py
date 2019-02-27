import numpy as np
from PIL import Image
import sys
import requests
from io import BytesIO
from skimage import color


def download_img(url, resize=None):
    img = Image.open(BytesIO(requests.get(url).content))
    if resize:
        if type(resize) == type(0.1) or type(resize) == type(1):
            img = img.resize(
                (int(img.width*resize), int(img.height*resize)), Image.ANTIALIAS)
        elif type(resize) == type((0, 1)):
            img = img.resize(resize, Image.ANTIALIAS)
    img = np.asarray(img)
    return img


def color_transfer(src,target):
    src = color.rgb2lab(src)
    target = color.rgb2lab(target)
    src_flat = np.reshape(src,(np.prod(src.shape[:-1]),src.shape[-1]))
    target_flat = np.reshape(target,(np.prod(target.shape[:-1]),target.shape[-1]))
    src_mean = np.mean(src_flat,axis=0)
    target_mean = np.mean(target_flat,axis=0)
    src_std = np.std(src_flat,axis=0)
    target_std = np.std(target_flat,axis=0)
    target_flat = target_flat - target_mean
    target_flat *= (target_std/src_std)
    target_flat += src_mean
    np.max(target_flat,axis=0)
    imgs = target_flat.reshape(target.shape)
    imgs = np.clip(imgs,-75,75)
    imgs = color.lab2rgb(imgs)
    return imgs
