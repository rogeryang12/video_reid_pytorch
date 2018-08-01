import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import Compose


__all__ =[
    'Compose', 'Resize', 'RandomCrop', 'RandomHorizontalFlip',
    'ToTensor', 'Normalize', 'CenterCrop', 'FiveCrop',
]


class Resize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.m = T.Resize(size, interpolation)

    def __call__(self, imgs):
        if len(imgs.shape) == 4:
            imgs = np.stack(self.resize_img(img) for img in imgs)
        else:
            imgs = self.resize_img(imgs)
        return imgs

    def resize_img(self, img):
        h, w, c = img.shape
        if c > 3:
            rgb = np.array(self.m(Image.fromarray(img[:, :, :3])))
            of = np.array(self.m(Image.fromarray(img[:, :, 3:])))
            return np.concatenate([rgb, of], axis=-1)
        else:
            return np.array(self.m(Image.fromarray(img)))


class RandomCrop(object):

    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, imgs):
        *_, h, w, c = imgs.shape
        i = random.randint(0, h - self.h)
        j = random.randint(0, w - self.w)
        return imgs[..., i:i+self.h, j:j+self.w, :]


class CenterCrop(object):

    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, img):
        h, w, c = img.shape
        i = int(round((h - self.h) / 2.))
        j = int(round((w - self.w) / 2.))
        return img[i:i+self.h, j:j+self.w]


class FiveCrop(object):

    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, img):
        h, w, c = img.shape
        tl = img[:self.h, :self.w]
        tr = img[:self.h:, w-self.w:]
        bl = img[h-self.h:, :self.w]
        br = img[h-self.h:, w-self.w:]
        center = CenterCrop((self.h, self.w))(img)
        return tl, tr, bl, br, center


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        if random.random() < self.p:
            return np.flip(imgs, axis=-2).copy()
        return imgs


class ToTensor(object):

    def __call__(self, imgs):
        if len(imgs.shape) == 4:
            order = (0, 3, 1, 2)
        else:
            order = (2, 0, 1)
        imgs = torch.from_numpy(imgs.transpose(order))
        if isinstance(imgs, torch.ByteTensor):
            return imgs.float().div(255)
        return imgs


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean).view(-1, 1, 1)
        self.std = torch.Tensor(std).view(-1, 1, 1)

    def __call__(self, imgs):
        c = imgs.size(-3)
        return (imgs - self.mean[:c]) / self.std[:c]
