import os
import sys 
import numpy as np
import SimpleITK as sitk

class Registered_BraTS_Case():
    
    def __init__(self, dicom_dir, resize_to, task1_dir=None):
        self.dicom_dir = dicom_dir
        self.resize_to = resize_to
        self.task1_dir = task1_dir
    
    @staticmethod
    def resample(image, ref_image):

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_image)
        resampler.SetInterpolator(sitk.sitkLinear)

        resampler.SetTransform(sitk.AffineTransform(image.GetDimension()))
        resampler.SetOutputSpacing(ref_image.GetSpacing())
        resampler.SetSize(ref_image.GetSize())
        resampler.SetOutputDirection(ref_image.GetDirection())
        resampler.SetOutputOrigin(ref_image.GetOrigin())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resamped_image = resampler.Execute(image)
        
        return resamped_image
    
    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def get_crop_bb(image):
        inside_value = 0
        outside_value = 255
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)
        return bounding_box
    
    @staticmethod
    def crop_with_bb(image, bounding_box):
        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
    
    @staticmethod
    def threshold_based_crop(image):
        """
        This function is copied from here: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/70_Data_Augmentation.ipynb
        
        Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
        usually air. Then crop the image using the foreground's axis aligned bounding box.
        Args:
            image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                     (the assumption underlying Otsu's method.)
        Return:
            Cropped image based on foreground's axis aligned bounding box.                                 
        """
        # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
        # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
        inside_value = 0
        outside_value = 255
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute( sitk.OtsuThreshold(image, inside_value, outside_value) )
        bounding_box = label_shape_filter.GetBoundingBox(outside_value)
        # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
        return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])

    @staticmethod
    def swap_image_axes(image, order):
        return sitk.PermuteAxes(image, order)

    @staticmethod
    def axis_swap_order(image):
        direction = np.array([int(round(d)) for d in image.GetDirection()])
        if np.all(direction == np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            return [0,1,2]
        elif np.all(direction == np.array([0, 0, -1, 1, 0, 0, 0, -1, 0])):
            return [2,0,1]
        elif np.all(direction == np.array([1, 0, 0, 0, 0, 1, 0, -1, 0])):
            return [0,2,1]
        else:
            print(list(direction))
            return [0,1,2]

    @staticmethod
    def maybe_flip_axes(image):
        direction = np.array([int(round(d)) for i,d in enumerate(image.GetDirection()) if i%4==0])
        flips = [bool(d == -1) for d in direction]
        return sitk.Flip(image, list(flips))
    
    @staticmethod
    def resize_image(original_CT, resize_to=[256,256,100]):

        dimension = original_CT.GetDimension()
        reference_physical_size = np.zeros(original_CT.GetDimension())
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

        reference_origin = original_CT.GetOrigin()
        reference_direction = original_CT.GetDirection()

        reference_size = resize_to
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

        reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(original_CT.GetDirection())

        transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)

        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform = sitk.Transform(centering_transform)

        return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)

    def get_registered_case(self, brat_id_str):
        """ Return stack of min-max normalized R:t1w, G:T1ce, B:T2 array """
        stacked = np.zeros(tuple([self.resize_to[2],*self.resize_to[:2],3]))
        try:
            # init sitk reader
            reader = sitk.ImageSeriesReader()
            reader.LoadPrivateTagsOn()
            
            # Use t1w as the reference and skip flair because it's sometimes completely dark
            filenamesDICOM = reader.GetGDCMSeriesFileNames(f'{self.dicom_dir}/{brat_id_str}/T1w')
            reader.SetFileNames(filenamesDICOM)
            t1w = reader.Execute()

            filenamesDICOM = reader.GetGDCMSeriesFileNames(f'{self.dicom_dir}/{brat_id_str}/T1wCE')
            reader.SetFileNames(filenamesDICOM)
            t1 = reader.Execute()

            filenamesDICOM = reader.GetGDCMSeriesFileNames(f'{self.dicom_dir}/{brat_id_str}/T2w')
            reader.SetFileNames(filenamesDICOM)
            t2 = reader.Execute()
            
            # Align reference image and crop with otsu to minimum enclosing box
            t1w = self.swap_image_axes(t1w, self.axis_swap_order(t1w))
            t1w = self.maybe_flip_axes(t1w)
            bounding_box = self.get_crop_bb(t1w)
            
            # resample other modalities to align with reference
            t1_resampled = self.resample(t1, t1w)
            t2_resampled = self.resample(t2, t1w)
            
            # crop all modalities to same box
            t1w_cropped = self.crop_with_bb(t1w, bounding_box)
            t1w_cropped = self.resize_image(t1w_cropped, self.resize_to)

            t1_cropped = self.crop_with_bb(t1_resampled, bounding_box)
            t1_cropped = self.resize_image(t1_cropped, self.resize_to)

            t2_cropped = self.crop_with_bb(t2_resampled, bounding_box)
            t2_cropped = self.resize_image(t2_cropped, self.resize_to)
            
            # Return stack of min-max normalized R:t1w, G:T1ce, B:T2 array
            stacked = np.stack([
                self.normalize(sitk.GetArrayFromImage(t1w_cropped)),
                self.normalize(sitk.GetArrayFromImage(t1_cropped)),
                self.normalize(sitk.GetArrayFromImage(t2_cropped))
            ], axis=3)
        except:
            print(f'Error: {brat_id_str}')
        return stacked
    
    def get_aligned_seg_map(self, brat_id_str):
        """ Normalizes task 1 segmentation map with similar logic and returns an aligned seg map """
        if self.task1_dir is None:
            return None
        
        try:
            segmentation_nii_fn = [os.path.join(f'{self.task1_dir}/BraTS2021_{brat_id_str}/', fn) for fn in os.listdir(f'{self.task1_dir}/BraTS2021_{brat_id_str}/') if 'seg' in fn]
            t1_nii_fn = [os.path.join(f'{self.task1_dir}/BraTS2021_{brat_id_str}/', fn) for fn in os.listdir(f'{self.task1_dir}/BraTS2021_{brat_id_str}/') if 't1.' in fn]

            if len(segmentation_nii_fn) == 0 or len(t1_nii_fn) ==0:
                return None

            t1 = sitk.ReadImage(t1_nii_fn[0])
            seg = sitk.ReadImage(segmentation_nii_fn[0])

            t1 = self.swap_image_axes(t1, self.axis_swap_order(t1))
            t1 = self.maybe_flip_axes(t1)
            bounding_box = self.get_crop_bb(t1)

            seg_resampled = self.resample(seg, t1)
            seg_cropped = self.crop_with_bb(seg_resampled, bounding_box)
            seg_cropped = self.resize_image(seg_cropped, self.resize_to)
        except:
            print(f'Seg error: {brat_id_str}')
            return None

        return sitk.GetArrayFromImage(seg_cropped)