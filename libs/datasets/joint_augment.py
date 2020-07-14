
import numpy as np
import random
import numbers
from PIL import Image, ImageEnhance, ImageOps
from .augment import to_pil_image
import torch
from torchvision.transforms import functional as F


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class To_Tensor():
    def __call__(self, arr, arr2):
        if len(np.array(arr).shape) == 2:
            arr = np.array(arr)[:,:,None]
        if len(np.array(arr2).shape) == 2:
            arr2 = np.array(arr2)[:, :, None]
        arr = torch.from_numpy(np.array(arr).transpose(2,0,1))
        arr2 = torch.from_numpy(np.array(arr2).transpose(2,0,1))
        return arr, arr2

class To_PIL_Image():
    def __call__(self, img, mask):
        img = to_pil_image(img)
        mask = to_pil_image(mask)
        return img, mask

class RandomVerticalFlip():
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=0), np.flip(mask, axis=0)
        return img, mask

class RandomHorizontallyFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=1), np.flip(mask, axis=1)
        return img, mask

class RandomRotate():
    def __init__(self, degrees, prob=0.5):
        self.prob = prob
        self.degrees = degrees

    def __call__(self, img, mask):
        if random.random() < self.prob:
            rotate_detree = random.uniform(self.degrees[0], self.degrees[1])
            return img.rotate(rotate_detree, Image.BILINEAR), mask.rotate(rotate_detree, Image.NEAREST)
        return img, mask

class FixResize():
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img=None, mask=None):
        if mask is None:
            return img.resize(self.size, Image.BILINEAR)
        if img is None:
            return mask.resize(self.size, Image.NEAREST)
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))

class ScaleRatio():
    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor
    
    def __call__(self, img, interpolation):
        w, h = img.size
        new_h = int(h * self.scale_factor)
        new_w = int(w * self.scale_factor)
        return img.resize((new_w, new_h), interpolation)

class RandomScale():
    def __init__(self, min_factor=0.8, max_factor=1.2, prob=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob

    def __scale(self, img, scale_factor, interpolation):
        w, h = img.size
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        return img.resize((new_w, new_h), interpolation)

    def __call__(self, img, mask):
        if random.random() < self.prob:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            return self.__scale(img, factor, Image.BILINEAR), self.__scale(mask, factor, Image.NEAREST)
        return img, mask


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size, Image.NEAREST)
        return image, target


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0, prob=0.5):
        self.prob = prob
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, mask):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        if random.random() < self.prob:
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
            return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor), F.affine(mask, *ret, resample=self.resample, fillcolor=self.fillcolor)
        return img, mask

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)
