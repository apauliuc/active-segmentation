import random
import numpy as np
from PIL.Image import Image, FLIP_LEFT_RIGHT
from PIL.Image import BILINEAR
from torchvision.transforms import transforms
from helpers.types import floatTensor, longTensor


class Rotation(object):
    """
    Rotates an image to random angle
    """

    def __init__(self) -> None:
        self._angles = np.arange(0, 5, 1)

    def __call__(self, sample: tuple) -> tuple:
        return self.rotate(sample[0], sample[1])

    def rotate(self, image: Image, segmentation: Image) -> tuple:
        """

        :param image: Image to be rotated at random angle
        :param segmentation: Segmentation rotated at random angle
        :return: Rotated PIL image and segmentation.
        """
        assert isinstance(image, Image) and isinstance(segmentation, Image)
        angle = random.choice(self._angles)

        return image.rotate(angle, resample=BILINEAR), segmentation.rotate(angle, resample=BILINEAR)


class ToPILImage(object):
    """
    Converts npy array to PIL Image
    """

    def __init__(self) -> None:
        self._to_pil = transforms.ToPILImage()

    def __call__(self, sample: tuple) -> tuple:
        return self.to_pil_image(sample[0], sample[1])

    def to_pil_image(self, image, segmentation) -> tuple:
        """

        :param image: Image to be converted to PIL Image.
        :param segmentation: Segmentation to be converted to PIL Image.
        :return: Converted PIL image and segmentation.
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if len(segmentation.shape) == 2:
            segmentation = np.expand_dims(segmentation, axis=2)

        return self._to_pil(image), self._to_pil(segmentation)


class ToTensor(object):
    """
    Convert npy array to tensor
    """

    def __init__(self) -> None:
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample: tuple) -> tuple:
        return self.toTensor(sample[0], sample[1])

    def toTensor(self, image, segmentation) -> tuple:
        """

        :param image: Image to be converted to tensor.
        :param segmentation: Segmentation to be converted to tensor.
        :return: Image and segmentation as tensor object
        """
        return self.to_tensor(image).type(floatTensor), self.to_tensor(segmentation).type(floatTensor)


class Flip(object):
    """
    Randomly flip an image and its segmentation horizontally
    """

    def __call__(self, sample: tuple) -> tuple:
        return self.flip(sample[0], sample[1])

    def flip(self, image: Image, segmentation: Image) -> tuple:
        """

        :param image: Image to be randomly flipped.
        :param segmentation: Segmentation to be randomly flipped.
        :return: Randomly flipped image and segmentation
        """
        assert isinstance(image, Image) and isinstance(segmentation, Image)

        p = random.random()

        if p < 0.5:
            return image, segmentation
        else:
            return image.transpose(FLIP_LEFT_RIGHT), segmentation.transpose(FLIP_LEFT_RIGHT)


class FlipNumpy(object):
    """
    Randomly flip image/segmentation as Numpy objects
    """

    def __call__(self, sample: tuple) -> tuple:
        return self.flip(sample[0], sample[1])

    def flip(self, image: np.array, segmentation: np.array) -> tuple:
        """

        :param image: image as NPY array to flip
        :param segmentation: segmentation as NPY array to flip
        :return: Randomly flipped image and segmentation
        """
        p = random.random()

        if p < 0.5:
            return image, segmentation
        else:
            return np.fliplr(image), np.fliplr(segmentation)
