import random

import numpy as np
from PIL.Image import FLIP_LEFT_RIGHT, BILINEAR


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, segmentation):
        assert image.size == segmentation.size

        for t in self.transforms:
            image, segmentation = t(image, segmentation)
        return image, segmentation

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlip(object):
    def __call__(self, image, segmentation):
        if random.random() < 0.5:
            return image.transpose(FLIP_LEFT_RIGHT), segmentation.transpose(FLIP_LEFT_RIGHT)
        return image, segmentation


class Rotate(object):
    """
    Rotates an image to random angle
    """

    def __init__(self):
        self._angles = np.arange(-15, 15, 1)

    def __call__(self, image, segmentation):
        angle = random.choice(self._angles)

        return image.rotate(angle, resample=BILINEAR), segmentation.rotate(angle, resample=BILINEAR)
