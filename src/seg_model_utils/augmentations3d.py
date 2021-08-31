import random
import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

class RandomChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        t = random.choice(self.transforms)
        return t(sample)
    
class ComposeTransforms(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms=[],
                 p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        for t in self.transforms:
            sample = t(sample)
        return sample

def stack_seg_2_image(sample):
    image = sample['image']
    seg = sample['segmentation']
    channels = [chan for chan in image]
    channels.append(seg)
    return np.stack(channels, axis=3)

def elastic_transform_3d(sample, alpha=1, sigma=20, c_val=0.0, method="linear"):
    """
    :param sample: dict of image and seg
    :param alpha: scaling factor of gaussian filter
    :param sigma: standard deviation of random gaussian filter
    :param c_val: fill value
    :param method: interpolation method. supported methods : ("linear", "nearest")
    :return: deformed image and/or label
    """
    img_numpy = sample['image'].copy()
    label = sample['segmentation'] if 'segmentation' in sample else None
    shape = img_numpy.shape
    
    # Define 3D coordinate system
    coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])

    # Interpolated images
    chan_intrps = [RegularGridInterpolator(coords, img_numpy[:,:,:,chan],
                                        method=method,
                                        bounds_error=False,
                                        fill_value=c_val) for chan in range(shape[3])]

    #Get random elastic deformations
    dx = gaussian_filter((np.random.rand(shape[0],shape[1],shape[2]) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dy = gaussian_filter((np.random.rand(shape[0],shape[1],shape[2]) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha
    dz = gaussian_filter((np.random.rand(shape[0],shape[1],shape[2]) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    indices = np.reshape(x + dx, (-1, 1)), \
              np.reshape(y + dy, (-1, 1)), \
              np.reshape(z + dz, (-1, 1))

    # Interpolate 3D image image
    img_numpy = np.stack([chan_intrp(indices).reshape((shape[0],shape[1],shape[2])) 
                          for chan_intrp in chan_intrps], axis=3).astype(np.float32)

    # Interpolate labels
    if label is not None:
        lab_intrp = RegularGridInterpolator(coords, label,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0)

        label = lab_intrp(indices).reshape(shape[0],shape[1],shape[2]).astype(label.dtype)
        sample['segmentation'] = label
    
    sample['image'] = img_numpy
    return sample


class ElasticTransform(object):
    def __init__(self, p=0.5, alpha=1, sigma=20, c_val=0.0, method="linear"):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        self.c_val = c_val
        self.method = method

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        return elastic_transform_3d(sample, self.alpha, self.sigma, self.c_val, self.method)
    
def random_noise(sample, mean=0, std=0.001, eps=1e-6):
    im = sample['image'].copy()
    noise = np.random.normal(mean, std, im.shape)
    sample['image'] = np.where(im > eps, im + noise, im)
    return sample


class GaussianNoise(object):
    def __init__(self, p=0.5, mean=0, std=0.001):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        
        return random_noise(sample, self.mean, self.std)
    
def random_crop_to_size(sample, crop_sz):
    
    im = sample['image'].copy()
    seg = sample['segmentation'].copy()
    shape = im.shape
    
    width, height, depth = crop_sz
    d = np.random.randint(0, shape[0] - depth - 1)
    x = np.random.randint(0, shape[1] - width - 1)
    y = np.random.randint(0, shape[2] - height - 1)
    
    im = im[d:d+depth, x:x+width, y:y+height,:]
    seg = seg[d:d+depth, x:x+width, y:y+height]
    sample['image'] = im
    sample['segmentation'] = seg
    
    return sample

class RandomCropToSize(object):
    
    def __init__(self, crop_sz=(200,200,95)):
        self.crop_sz = crop_sz

    def __call__(self, sample):
        return random_crop_to_size(sample, self.crop_sz)
    
def random_flip_lr(sample):
    im = sample['image'].copy()
    seg = sample['segmentation'].copy()
    im = im[:,:,::-1,:]
    seg = seg[:,:,::-1]
    
    sample['image'] = im
    sample['segmentation'] = seg
    
    return sample

class RandomFlipLR(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        return random_flip_lr(sample)

def random_channel_drop(sample):
    im = sample['image'].copy()
    c = im.shape[3]
    drop_ch = random.randint(0, c-1)
    im[:,:,:,drop_ch] = 0. if random.random() > 0.5 else 1.0
    sample['image'] = im
    return sample
    
class RandomChannelDrop(object):
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        return random_channel_drop(sample)

def random_rotate3D(sample, min_angle, max_angle):
    """
    Returns a random rotated image and seg map in sample dict
    :param sample: ds sample dict
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: sample
    """
    im = sample['image'].copy()
    seg = sample['segmentation'].copy()
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    
    im = ndimage.interpolation.rotate(im , angle, axes=axes, reshape=False)
    
    seg = ndimage.rotate(seg.astype(np.float32), angle, axes=axes, reshape=False)
    
    # seg back to binary float values
    seg = np.where(seg < 0.5, 0, 1.)
    
    sample['image'] = im
    sample['segmentation'] = seg
    
    return sample


class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10, p=0.5):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample):
        augment = np.random.random(1) < self.p
        if not augment:
            return sample
        return random_rotate3D(sample, self.min_angle, self.max_angle)