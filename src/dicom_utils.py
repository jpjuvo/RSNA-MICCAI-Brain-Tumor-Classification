import numpy as np
import cv2

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

__all__ = [
    "get_clahe",
    "get_uint8_rgb"
]

def _read_dicom_image(
    path,
    voi_lut=True,
    fix_monochrome=True,
    do_norm=True):

    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.max(data) - data

    if do_norm:
        data = data - np.min(data)
        data = (data / np.max(data)) if np.max(data) > 0 else data
    
    return data.astype(np.float32)

def get_clahe():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))

def _calc_image_features(image, clahe=None):
    # from https://www.kaggle.com/socom20/effdet-v2
    img_uint = (255 * image).astype(np.uint8)
    clahe = clahe or get_clahe()

    try:
        img_equ  = cv2.equalizeHist(img_uint) if np.max(img_uint) > 0 else img_uint
        img_clahe = clahe.apply(img_uint) if np.max(img_uint) > 0 else img_uint

        img_ret = np.concatenate(
            [
                image[:,:,None],
                img_clahe[:,:,None].astype(np.float32)  / 255,
                img_equ[:,:,None].astype(np.float32)  / 255,
                ],
            axis=-1)
    except:
        print('exception')
        img_ret = np.concatenate(
            [
                image[:,:,None],
                image[:,:,None],
                image[:,:,None],
                ],
            axis=-1)
        
    return img_ret

def get_uint8_rgb(dicom_path):
    """ 
    Reads dicom from path and returns rgb uint8 array
    where R: min-max normalized, G: CLAHE, B: histogram equalized.
    Image size remains original. 
    """
    dcm = _read_dicom_image(dicom_path)
    feats = _calc_image_features(dcm)
    return (feats*255).astype(np.uint8)