import cv2
import random
import functools 
from typing import Union
from PIL import Image
from collections.abc import Sequence
from imgaug import augmenters as iaa
import numpy as np

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from pyiqa.archs.arch_util import to_2tuple


def transform_mapping(key, args):
    if key == 'hflip' and args:
        return [PairedRandomHorizontalFlip()] 
    if key == 'vflip' and args:
        return [PairedRandomVerticalFlip()] 
    elif key == 'random_crop':
        return [PairedRandomCrop(args)]
    elif key == 'center_crop':
        return [PairedCenterCrop(args)]
    elif key == 'resize':
        return [PairedResize(args)]
    elif key == 'adaptive_resize':
        return [PairedAdaptiveResize(args)]
    elif key == 'random_square_resize':
        return [PairedRandomSquareResize(args)]
    elif key == 'random_arp_resize':
        return [PairedRandomARPResize(args)]
    elif key == 'ada_pad':
        return [PairedAdaptivePadding(args)]
    elif key == 'rot90' and args:
        return [PairedRandomRot90(args)]
    elif key == 'randomerase':
        return [PairedRandomErasing(**args)]
    elif key == 'changecolor':
        return [ChangeColorSpace(args)]
    elif key == 'totensor' and args:
        return [PairedToTensor()]
    else:
        return []


def _is_pair(x):
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        return True


class PairedToTensor(tf.ToTensor):
    """Pair version of center crop"""
    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return super().__call__(x)

    def __call__(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = self.to_tensor(imgs[i])
            return imgs 
        else:
            return self.to_tensor(imgs) 


class ChangeColorSpace:
    """Pair version of center crop"""
    def __init__(self, to_colorspace):
        self.aug_op = iaa.color.ChangeColorspace(to_colorspace)

    def __call__(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                tmpimg = self.aug_op.augment_image(np.array(imgs[i]))
                imgs[i] = Image.fromarray(tmpimg)
            return imgs 
        else:
            imgs = self.aug_op.augment_image(np.array(imgs))
            return Image.fromarray(imgs)


class PairedCenterCrop(tf.CenterCrop):
    """Pair version of center crop"""
    def forward(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = super().forward(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomCrop(tf.RandomCrop):
    """Pair version of random crop"""
    def _pad(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = img.size 
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        return img

    def forward(self, imgs):
        if _is_pair(imgs):
            i, j, h, w = self.get_params(imgs[0], self.size)
            for i in range(len(imgs)):
                img = self._pad(imgs[i]) 
                img = F.crop(img, i, j, h, w)
                imgs[i] = img
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomErasing(tf.RandomErasing):
    """Pair version of random erasing"""

    def forward(self, imgs):
        if _is_pair(imgs):
            if torch.rand(1) < self.p:
                # cast self.value to script acceptable type
                if isinstance(self.value, (int, float)):
                    value = [self.value]
                elif isinstance(self.value, str):
                    value = None
                elif isinstance(self.value, tuple):
                    value = list(self.value)
                else:
                    value = self.value

                if value is not None and not (len(value) in (1, imgs[0].shape[-3])):
                    raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        f"{imgs[0].shape[-3]} (number of input channels)"
                    )

                x, y, h, w, v = self.get_params(imgs[0], scale=self.scale, ratio=self.ratio, value=value)
                for i in range(len(imgs)):
                    imgs[i] = F.erase(imgs[i], x, y, h, w, v, self.inplace)
            return imgs 
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomHorizontalFlip(tf.RandomHorizontalFlip):
    """Pair version of random hflip"""
    def forward(self, imgs):
        if _is_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.hflip(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)


class PairedRandomVerticalFlip(tf.RandomVerticalFlip):
    """Pair version of random hflip"""
    def forward(self, imgs):
        if _is_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.vflip(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image):
            return super().forward(imgs)



class PairedRandomRot90(torch.nn.Module):
    """Pair version of random hflip"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if _is_pair(imgs):
            if torch.rand(1) < self.p:
                for i in range(len(imgs)):
                    imgs[i] = F.rotate(imgs[i], 90)
            return imgs
        elif isinstance(imgs, Image.Image):
            if torch.rand(1) < self.p:
                imgs = F.rotate(imgs, 90)
            return imgs


class PairedResize(tf.Resize):
    """Pair version of resize"""
    def forward(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = super().forward(imgs[i])
            return imgs
        elif isinstance(imgs, Image.Image): 
            return super().forward(imgs)


class PairedAdaptiveResize(tf.Resize):
    """ARP preserved resize when necessary"""
    def forward(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                tmpimg = imgs[i]
                min_size = min(tmpimg.size)
                if min_size < self.size:
                    tmpimg = super().forward(tmpimg)
                imgs[i] = tmpimg
            return imgs
        elif isinstance(imgs, Image.Image): 
            tmpimg = imgs
            min_size = min(tmpimg.size)
            if min_size < self.size:
                tmpimg = super().forward(tmpimg)
            return tmpimg


class PairedRandomARPResize(torch.nn.Module):
    """Pair version of resize"""
    def __init__(self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(f"size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}")

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image): 
            return F.resize(imgs, target_size, self.interpolation)


class PairedRandomSquareResize(torch.nn.Module):
    """Pair version of resize"""
    def __init__(self, size_range, interpolation=tf.InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size_range = size_range
        if not (isinstance(size_range, Sequence) and len(size_range) == 2):
            raise TypeError(f"size_range should be sequence with 2 int. Got {size_range} with {type(size_range)}")

    def forward(self, imgs):
        min_size, max_size = sorted(self.size_range)
        target_size = random.randint(min_size, max_size)
        target_size = (target_size, target_size)
        if _is_pair(imgs):
            for i in range(len(imgs)):
                imgs[i] = F.resize(imgs[i], target_size, self.interpolation)
            return imgs
        elif isinstance(imgs, Image.Image): 
            return F.resize(imgs, target_size, self.interpolation)


class PairedAdaptivePadding(torch.nn.Module):
    """Pair version of resize"""
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        super().__init__()
        self.target_size = to_2tuple(target_size)
        self.fill = fill
        self.padding_mode = padding_mode
    
    def get_padding(self, x):
        w, h = x.size
        th, tw = self.target_size
        assert th >= h and tw >= w, f'Target size {self.target_size} should be larger than image size ({h}, {w})'
        pad_row = th - h 
        pad_col = tw - w
        pad_l, pad_r, pad_t, pad_b = (pad_col//2, pad_col - pad_col//2, pad_row//2, pad_row - pad_row//2)
        return (pad_l, pad_t, pad_r, pad_b)

    def forward(self, imgs):
        if _is_pair(imgs):
            for i in range(len(imgs)):
                padding = self.get_padding(imgs[i])
                imgs[i] = F.pad(imgs[i], padding, self.fill, self.padding_mode)
            return imgs
        elif isinstance(imgs, Image.Image): 
            padding = self.get_padding(imgs)
            imgs = F.pad(imgs, padding, self.fill, self.padding_mode)
            return imgs


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img



def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
