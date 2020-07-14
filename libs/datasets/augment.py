
import numpy as np
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class to_Tensor():
    def __call__(self,arr):
        if len(np.array(arr).shape) == 2:
            arr = np.array(arr)[:,:,None]
        arr = torch.from_numpy(np.array(arr).transpose(2,0,1))
        return arr

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')
    return im.resize(size, resample)

class To_PIL_Image():
    def __call__(self, img):
        return to_pil_image(img)

class normalize():
    def __init__(self,mean,std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def __call__(self,img):
        self.mean = torch.as_tensor(self.mean,dtype=img.dtype,device=img.device)
        self.std = torch.as_tensor(self.std,dtype=img.dtype,device=img.device)
        return (img-self.mean)/self.std

class RandomVerticalFlip():
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=0)
        return img

class RandomHorizontallyFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(img, np.ndarray):
                return np.flip(img, axis=1)
        return img

class RandomRotate():
    def __init__(self, degree, prob=0.5):
        self.prob = prob
        self.degree = degree

    def __call__(self, img, interpolation=Image.BILINEAR):
        if random.random() < self.prob:
            rotate_detree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_detree, interpolation)
        return img

class RandomBrightness():
    def __init__(self, min_factor, max_factor, prob=0.5):
        """ :param min_factor: The value between 0.0 and max_factor
            that define the minimum adjustment of image brightness.
            The value  0.0 gives a black image,The value 1.0 gives the original image, value bigger than 1.0 gives more bright image.
            :param max_factor: A value should be bigger than min_factor.
            that define the maximum adjustment of image brightness.
            The value  0.0 gives a black image, value 1.0 gives the original image, value bigger than 1.0 gives more bright image.

        """
        self.prob = prob
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    # def __brightness(self, img, factor):
    #     return img * (1.0 - factor) + img * factor

    # def __call__(self, img):
    #     if random.random() < self.prob:
    #         factor = np.random.uniform(self.min_factor, self.max_factor)
    #         return self.__brightness(img, factor)

    def __call__(self, img):
        if random.random() < self.prob:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            enhancer_brightness = ImageEnhance.Brightness(img)
            return enhancer_brightness.enhance(factor)

        return img

class RandomContrast():
    def __init__(self, min_factor, max_factor, prob=0.5):
        """ :param min_factor: The value between 0.0 and max_factor
            that define the minimum adjustment of image contrast.
            The value  0.0 gives s solid grey image, value 1.0 gives the original image.
            :param max_factor: A value should be bigger than min_factor.
            that define the maximum adjustment of image contrast.
            The value  0.0 gives s solid grey image, value 1.0 gives the original image.
        """
        self.prob = prob
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        if random.random() < self.prob:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            enhance_contrast = ImageEnhance.Contrast(img)
            return enhance_contrast.enhance(factor)
        return img

def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    # if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
    #     raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)
