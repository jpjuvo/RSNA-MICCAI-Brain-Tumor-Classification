import torch
import torchio as tio
import numpy as np

def load_tio_image(fn):
    """
    ScalarImage(shape: (c, w, h, d))
    dtype: torch.DoubleTensor
    """
    arr = np.load(fn).swapaxes(0,3)
    return tio.ScalarImage(tensor=arr)

def arr_2_tio_image(arr):
    """
    ScalarImage(shape: (c, w, h, d))
    dtype: torch.DoubleTensor
    """
    arr = arr.swapaxes(0,3)
    return tio.ScalarImage(tensor=arr)

def load_tio_seg_image(fn):
    """
    LabelMap(shape: (c, w, h, d))
    dtype: torch.FloatTensor
    
    Intensity transforms are not applied to these images.
    Nearest neighbor interpolation is always used to resample label maps.
    """
    if fn is None:
        return None
    if not os.path.exists(fn):
        return None
    arr = (np.expand_dims(np.load(fn),3).swapaxes(0,3) > 0).astype(np.float32)
    return tio.LabelMap(tensor=arr)

def arr_2_tio_seg_image(arr):
    """
    LabelMap(shape: (c, w, h, d))
    dtype: torch.FloatTensor
    
    Intensity transforms are not applied to these images.
    Nearest neighbor interpolation is always used to resample label maps.
    """
    if arr is None:
        return None
    arr = (np.expand_dims(arr,3).swapaxes(0,3) > 0).astype(np.float32)
    return tio.LabelMap(tensor=arr)

def load_tio_subject(image_fn:str, label:int, seg_fn=None):
    return tio.Subject(
        rgb_image=load_tio_image(image_fn),
        segmentation=load_tio_seg_image(seg_fn),
        label=int(label),
        name=os.path.basename(image_fn).split('.')[0])