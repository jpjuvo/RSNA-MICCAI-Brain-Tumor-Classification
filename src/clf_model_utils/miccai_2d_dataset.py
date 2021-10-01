import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset
import cv2
import SimpleITK as sitk
import fastai
from fastai.vision.all import *

class MICCAI2DDataset(Dataset):
    """ 
    Dataset that aligns T2W stack and normalizes slice thickness to 1.
    Axial 2D images are MIPed (max. intensity projection) from tumor region height.
    """

    def __init__(self, 
                 df_features,
                 image_dir=None,
                 npy_dir=None,
                 image_size=(256,256),
                 tio_augmentations=None,
                 is_train=True,
                 mip_window=0.1 # maximum intensity pooling for height slicing
                ):
        
        self.image_size = image_size
        self.image_dir = image_dir
        self.npy_dir = npy_dir
        self.df_features = df_features
        self.tio_augmentations = tio_augmentations
        self.is_train = is_train
        self.mip_window = mip_window
        
        # We use ToCanonical to have a consistent orientation, Resample the images to 1 mm isotropic spacing
        preprocessing_transforms = (
            tio.ToCanonical(),
            tio.Resample(1, image_interpolation='bspline'),
        )
        self.preprocess = tio.Compose(preprocessing_transforms)

        if is_train:
            # shuffle
            self.df_features = self.df_features.sample(frac=1)
            
            # method for placing oversampling but not implemented in base dataset
            self._sample_data()

    def _sample_data(self):
        pass
    
    @contextmanager
    def set_split_idx(self, i):
        """ Used by fastai's tta, when activating test time augs """
        if i == 0:
            self.tio_augmentations = tio.Compose([
                tio.RandomAffine(p=0.5),
                tio.RandomFlip(axes=(1,2), p=0.5)
            ])
        for _ in range(8):
            pass
        try: yield self
        finally:
            pass
            

    @staticmethod
    def _normalize(image, min_arr, max_arr):
        """ To [-1,1] range """
        image = (image.astype("float32", copy=False) - min_arr) / (max_arr - min_arr + 1e-6)
        image = image * 2 - 1
        return image
    
    def _resize(self, image):
        image = cv2.resize(image, self.image_size, cv2.INTER_LINEAR)
        return image
    
    def _get_crop_bb(self, image):
        inside_value = 0
        outside_value = 255
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)
        return bounding_box
    
    def _crop_with_bb(self, image, bounding_box):
        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    
    def _crop_tio_image(self, tio_image):
        sitk_image = tio_image.as_sitk()
        sitk_image = self._crop_with_bb(sitk_image, self._get_crop_bb(sitk_image))
        arr = sitk.GetArrayFromImage(sitk_image)
        arr = np.swapaxes(arr, 0,2)
        return tio.ScalarImage(tensor=np.expand_dims(arr, axis=0))
    
    def _extract_tumor_height(self, arr, features):
        """ MIP tumor slice """
        min_tumor_height = features['percentile10_ax_2'] / 100
        max_tumor_height = features['percentile90_ax_2'] / 100
        max_tumor_height -= self.mip_window
        if features['percentile10_ax_2'] == features['percentile90_ax_2']:
            tumor_height = 0.5
        elif min_tumor_height >= max_tumor_height:
            tumor_height = min(min_tumor_height, 1. - self.mip_window)
        else:
            tumor_height = np.random.random() * (max_tumor_height - min_tumor_height) + min_tumor_height
        tumor_height_start = int(tumor_height*arr.shape[3])
        tumor_height_end = int((tumor_height + self.mip_window)*arr.shape[3])
        
        return np.max(arr[:,:,:,tumor_height_start:tumor_height_end], axis=3)

    def __len__(self):
        return len(self.df_features)
    
    def _create_label(self, row):
        return float(row.MGMT_value)
    
    def __getitem__(self, idx):
        row = self.df_features.iloc[idx]
        bratsid = f'{int(row.BraTS21ID):05d}'
        
        # load image from preprocessed numpy or from dicom
        if self.npy_dir is not None:
            crop_image = tio.ScalarImage(tensor=np.load(os.path.join(self.npy_dir, f'{bratsid}.npy')))
        else:
            tio_image = tio.ScalarImage(os.path.join(self.image_dir, bratsid, 'T2w'))
            tio_image = self.preprocess.apply_transform(tio_image)
            crop_image = self._crop_tio_image(tio_image)
        
        # get min and max values before slicing - this way the normalization will maintain global range better
        tio_arr = crop_image.numpy().astype(np.float32)
        max_arr = np.max(tio_arr)
        min_arr = np.min(tio_arr)
        
        if self.tio_augmentations is not None:
            crop_image = self.tio_augmentations(crop_image)
        
        image = self._extract_tumor_height(crop_image.numpy(), row)
        image = image.astype(np.float32)
        
        # reduce one dimension in case it's (1,128,281) instead of (128,281)
        if len(image.shape) == 3:
            image = image[0]
            
        # resize
        image = self._resize(image)
        # normalize each patient
        image = self._normalize(image, min_arr, max_arr)
        
        # to 3chan rgb
        channels = [image, image, image]
        image = np.stack(channels)
        
        label = self._create_label(row)
        return image, label