import types
import random
import torch
import numpy as np

from PIL import Image, ImageFilter
from torchvision.transforms import functional as F


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, segmentation):
        assert image.size == segmentation.size

        for t in self.transforms:
            image, segmentation = t(image, segmentation)
        return image, segmentation


class RandomHorizontalFlip(object):
    def __call__(self, image, segmentation):
        if random.random() < 0.5:
            return image.transpose(Image.FLIP_LEFT_RIGHT), segmentation.transpose(Image.FLIP_LEFT_RIGHT)
        return image, segmentation


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, segmentation):
        assert image.size == segmentation.size
        w, h = image.size

        if (w >= h and w == self.size[0]) or (h >= w and h == self.size[1]):
            return image, segmentation

        return image.resize(self.size, Image.BILINEAR), segmentation.resize(self.size, Image.NEAREST)


class SegmentationToTensor(object):
    def __call__(self, segmentation):
        return torch.from_numpy(np.array(segmentation, dtype=np.int32)).long()


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return image


class DeNormalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class RandomGaussianBlur(object):
    def __call__(self, image, segmentation):
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, segmentation


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        return transforms

    @staticmethod
    def forward_transforms(image, transforms):
        for transform in transforms:
            image = transform(image)

        return image

    def __call__(self, image, segmentation):
        """
        Args:
            image (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transforms = self.get_params(self.brightness, self.contrast,
                                     self.saturation, self.hue)

        return self.forward_transforms(image, transforms), segmentation
