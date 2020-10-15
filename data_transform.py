import torchvision.transforms.functional as F
import random
import torch
import numpy as np
from PIL import Image


class CenterCrop(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, mask):

        # image transform
        image_0 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 0]), output_size=self.size)), axis=2)
        image_1 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 1]), output_size=self.size)), axis=2)
        image_2 = np.expand_dims(np.array(F.center_crop(img=Image.fromarray(image[:, :, 2]), output_size=self.size)), axis=2)
        image = np.concatenate((image_0, image_1, image_2), axis=2)

        # mask transform
        mask = np.array(F.center_crop(img=Image.fromarray(mask), output_size=self.size))

        return image, mask


class HorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            # image transform
            image_0 = np.expand_dims(np.array(F.hflip(Image.fromarray(image[:, :, 0]))), axis=2)
            image_1 = np.expand_dims(np.array(F.hflip(Image.fromarray(image[:, :, 1]))), axis=2)
            image_2 = np.expand_dims(np.array(F.hflip(Image.fromarray(image[:, :, 2]))), axis=2)
            image = np.concatenate((image_0, image_1, image_2), axis=2)

            # mask transform
            mask = np.array(F.hflip(Image.fromarray(mask)))

        return image, mask


class VerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:

            # image transform
            image_0 = np.expand_dims(np.array(F.vflip(Image.fromarray(image[:, :, 0]))), axis=2)
            image_1 = np.expand_dims(np.array(F.vflip(Image.fromarray(image[:, :, 1]))), axis=2)
            image_2 = np.expand_dims(np.array(F.vflip(Image.fromarray(image[:, :, 2]))), axis=2)
            image = np.concatenate((image_0, image_1, image_2), axis=2)
            # mask transform
            mask = np.array(F.vflip(Image.fromarray(mask)))

        return image, mask


class Rotate(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):

        angle = random.choice(self.degrees)

        # image transform
        image_0 = np.expand_dims(np.array(Image.fromarray(image[:, :, 0]).rotate(angle)), axis=2)
        image_1 = np.expand_dims(np.array(Image.fromarray(image[:, :, 1]).rotate(angle)), axis=2)
        image_2 = np.expand_dims(np.array(Image.fromarray(image[:, :, 2]).rotate(angle)), axis=2)

        image = np.concatenate((image_0, image_1, image_2), axis=2)

        # mask transform
        mask = np.array(Image.fromarray(mask).rotate(angle))

        return image, mask


class ToTensor(object):

    def __call__(self, image, mask):

        # image transform
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i]) - np.min(image[:, :, i]))
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # mask transform
        mask = torch.from_numpy(np.expand_dims(mask, axis=2).transpose((2, 0, 1)))

        return image, mask


class Normalize(object):

    def __call__(self, image, mask):

        # image transform
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask

