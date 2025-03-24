import random
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose, ToTensor, ToPILImage
from torch import Tensor
from typing import List, Union, Tuple, Dict
from PIL import Image
from drop.utils.misc import removeOmegaConf


# Define the custom transformation for Min-Max scaling
class MinMaxScale(object):
    def __init__(self, min_value, data_range):
        self.min_value = min_value
        self.data_range = data_range

    def __call__(self, tensor):
        return (tensor - self.min_value) / self.data_range


class CenterCropPIL:
    """Crop PIl image with dimensions (c, w, h). Dimnesions are flipped, cause we use a torch transformation.
    And return cropped PIL Image"""

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def __call__(self, img: PIL.Image) -> PIL.Image:
        img = CenterCrop(size=(self.h, self.w))(img)
        return img


class RandomCombination(Compose):
    """
    Apply each transformation in a list of transformations
    with a given probability.
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomCombination, self).__init__(transforms)
        self.p = p

    def __call__(self, img):
        for t in self.transforms:
            if self.p >= random.random():
                img = t(img)
        return img


class RandomOrthogonalRotate:
    """
    Randomly Tensor or PILImage rotate by 90, 180, or 270 degrees.
    """

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)


class HorizontalFlip:
    def __call__(self, img):
        return TF.hflip(img)


class VerticalFlip:
    def __call__(self, img):
        return TF.vflip(img)


class GaussianNoise:
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, img: PIL.Image) -> PIL.Image:
        img = ToTensor()(img)
        noise = torch.ones(img.shape)
        img = img + torch.normal(mean=noise * 0, std=noise * self.sigma)
        img = torch.clamp(img, 0, 1)  # maybe delete
        return ToPILImage()(img)


class Normalize:
    """Every channel is normalised to have specfified mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):

        img = transforms.Normalize(self.mean, self.std)(img)

        return img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tens: Tensor):
        """
        Params:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tens, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tens


class NormalizeMinMax:
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-8

    def __call__(self, img: Tensor) -> Tensor:
        # img : BCHW
        imin = torch.amin(img, dim=(1, 2, 3), keepdim=True)
        imax = torch.amax(img, dim=(1, 2, 3), keepdim=True)
        print(imin, imax)
        # # Put into [0, 1] - should already be actually
        img = (img - imin) / (imax - imin + self.eps)
        # Put into [pmin, pmax]
        img = (img * (self.pmax - self.pmin)) + self.pmin
        return img


class RandomShapeAugment:
    def __call__(self, img: PIL.Image):
        img = RandomCombination([RandomOrthogonalRotate(), HorizontalFlip(), VerticalFlip()])(img)
        return img


class RandomColorAugmentation:
    # is a ListConfig to be exact. but then cannot do typing.
    def __init__(
        self,
        brightness_jitter_amount: Union[List[float], float] = 0.0,
        contrast_jitter_amount: Union[List[float], float] = 0.0,
        saturation_jitter_amount: Union[List[float], float] = 0.0,
        hue_jitter_amount: Union[List[float], float] = 0.0,
        gaussian_noise_amount: float = 0.0,
        augmentation_probability: float = 0.5,
    ):
        self.brightness_jitter_amount = removeOmegaConf(brightness_jitter_amount)
        self.contrast_jitter_amount = removeOmegaConf(contrast_jitter_amount)
        self.saturation_jitter_amount = removeOmegaConf(saturation_jitter_amount)
        self.hue_jitter_amount = removeOmegaConf(hue_jitter_amount)
        self.gaussian_noise_amount = removeOmegaConf(gaussian_noise_amount)
        self.augmentation_probability = removeOmegaConf(augmentation_probability)

    def __call__(self, img: PIL.Image):
        """
        For each image quality (brightness, noise, etc.) an augmentation is applied with augmentation_probability.
        If amounts are a single value:
            The amount of the image quality changed is uniformly sampled between [(1-amount), (1+amount)].
        If amounts are an interval:
            The amount of the image quality changed is uniformly sampled within the interval specified.
        """
        img = RandomCombination(
            transforms=[
                transforms.ColorJitter(brightness=self.brightness_jitter_amount),
                transforms.ColorJitter(contrast=self.contrast_jitter_amount),
                transforms.ColorJitter(saturation=self.saturation_jitter_amount),
                transforms.ColorJitter(hue=self.hue_jitter_amount),
                GaussianNoise(sigma=self.gaussian_noise_amount),
            ],
            p=self.augmentation_probability,
        )(img)
        return img


class ReshapeImage:
    def __init__(self, target_image_size: Tuple[int, int]) -> None:
        """
        PIL (c,w,h) and pytorch (c,h,w) use inverse dimension ordering conventions
        Alternative is to use torch and nearest interpolation mode.

        Params
        _____
        target_image_size: Tuple[int, int]
            Expects value as [width, height].
        """
        self.target_w = target_image_size[0]
        self.target_h = target_image_size[1]

    def __call__(self, img: PIL.Image):
        img = img.resize((self.target_w, self.target_h), resample=Image.BICUBIC)
        return img


class ZeroPadToSquarePIL:
    def __init__(self):
        pass

    def __call__(self, img: PIL.Image):
        # determine smaller and larger side of the image.

        max_idx = np.argmin(img.size)
        min_idx = 1 - max_idx

        if img.size[max_idx] != img.size[min_idx]:
            padding_size = (img.size[max_idx] - img.size[min_idx]) // 2

            if max_idx == 0:
                padding = (0, padding_size, 0, padding_size)

            elif max_idx == 1:
                padding = (padding_size, 0, padding_size, 0)

            img = PIL.ImageOps.expand(img, padding)

        return img


class ZeroPadTensor:
    def __init__(self, size: Union[List, Tuple, int]):
        if isinstance(size, int):
            size = [size, size]

        self.size = size  # size should be in h,w configuration!!!

    def __call__(self, tens: Tensor):
        h, w = tens.shape[-2:]
        left = (self.size[1] - w) // 2
        right = self.size[1] - left - w
        top = (self.size[0] - h) // 2
        bottom = self.size[0] - top - h

        tens_padded = F.pad(tens, [left, right, top, bottom], mode="constant", value=0)

        return tens_padded
