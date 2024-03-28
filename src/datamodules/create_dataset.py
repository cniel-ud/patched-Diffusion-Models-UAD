import pickle
from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import torchio as tio
import os

sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager
from nilearn.image import crop_img, load_img
from nilearn.image.image import _crop_img_to
import nibabel as nib
import numpy as np
def load_splits(split_file):
    with open(split_file, 'rb') as f:
        split_info = pickle.load(f)
    return split_info


def Train(csv, cfg, preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'path': sub.img_path
        }
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
        else:  # if we don't have masks, we create a mask from the image
            subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)

    if preload:
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment=get_augment(cfg))
    else:
        ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))
    if cfg.spatialDims == '2D':
        slice_ind = cfg.get('startslice', None)
        seq_slices = cfg.get('sequentialslices', None)
        ds = vol2slice(ds, cfg, slice=slice_ind, seq_slices=seq_slices)
    return ds


def Eval(csv, cfg):
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path, reader=sitk_reader).shape != tio.ScalarImage(
                sub.mask_path, reader=sitk_reader).shape:
            print(
                f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path, reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path, reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')

        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            # we need the image in original size for evaluation
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'seg_available': False,
            'path': sub.img_path}
        if sub.seg_path != None:  # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path, reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,
                                                    reader=sitk_reader)  # we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,
                                                     reader=sitk_reader)  # we need the image in original size for evaluation
        else:
            tens = tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
    return ds


## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)


class preload_wrapper(Dataset):
    def __init__(self, ds, cache, augment=None):
        self.cache = cache
        self.ds = ds
        self.augment = augment

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if self.cache.is_cached(index):
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject


class vol2slice(Dataset):
    def __init__(self, ds, cfg, onlyBrain=False, slice=None, seq_slices=None):
        self.ds = ds
        self.onlyBrain = onlyBrain
        self.slice = slice
        self.seq_slices = seq_slices
        self.counter = 0
        self.ind = None
        self.cfg = cfg

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        if self.onlyBrain:
            start_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                if subject['mask'].data[0, :, :, i].any() and start_ind is None:  # only do this once
                    start_ind = i
                if not subject['mask'].data[0, :, :,
                       i].any() and start_ind is not None:  # only do this when start_ind is set
                    stop_ind = i
            low = start_ind
            high = stop_ind
        else:
            low = 0
            high = subject['vol'].data.shape[-1]
        if self.slice is not None:
            self.ind = self.slice
            if self.seq_slices is not None:
                low = self.ind
                high = self.ind + self.seq_slices
                self.ind = torch.randint(low, high, size=[1])
        else:
            if self.cfg.get('unique_slice', False):  # if all slices in one batch need to be at the same location
                if self.counter % self.cfg.batch_size == 0 or self.ind is None:  # only change the index when changing to new batch
                    self.ind = torch.randint(low, high, size=[1])
                self.counter = self.counter + 1
            else:
                self.ind = torch.randint(low, high, size=[1])

        subject['ind'] = self.ind

        subject['vol'].data = subject['vol'].data[..., self.ind]
        subject['mask'].data = subject['mask'].data[..., self.ind]

        return subject


def get_transform(cfg):  # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))

    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig']
    else:
        exclude_from_resampling = None

    if cfg.get('unisotropic_sampling', True):
        preprocess = tio.Compose([
            # tio.CropOrPad((h, w, d), padding_mode=0),
            tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                 masking_method='mask'),
            # tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # ,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else:
        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                                 masking_method='mask'),
            tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
            # ,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    return preprocess


def get_augment(cfg):  # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    augment = tio.Compose(augmentations)
    return augment


def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None


def exclude_empty_slices(image, mask=None, slice_dim=-1):
    slices = []
    mask_slices = []
    if slice_dim == -1:
        for i in range(image.shape[slice_dim]):
            if (image[..., i] > .0001).mean() >= .05:
                slices.append(image[..., i])
                if mask is not None:
                    mask_slices.append(mask[..., i])
    else:
        raise NotImplementedError(f'slice_dim = {slice_dim} is not supported')
    if mask is not None:
        return torch.tensor(slices).permute((1, 2, 0)), torch.tensor(mask_slices).permute((1, 2, 0))
    else:
        return torch.tensor(slices).permute((1, 2, 0))


def crop(img, mask=None, rtol=1e-8, copy=True, pad=True, return_offset=False):
    """Crops an image as much as possible.

    Will crop `img`, removing as many zero entries as possible without
    touching non-zero entries. Will leave one voxel of zero padding
    around the obtained non-zero area in order to avoid sampling issues
    later on.

    Parameters
    ----------
    img : Niimg-like object
        Image to be cropped (see :ref:`extracting_data` for a detailed
        description of the valid input types).

    rtol : :obj:`float`, optional
        relative tolerance (with respect to maximal absolute value of the
        image), under which values are considered negligeable and thus
        croppable. Default=1e-8.

    copy : :obj:`bool`, optional
        Specifies whether cropped data is copied or not. Default=True.

    pad : :obj:`bool`, optional
        Toggles adding 1-voxel of 0s around the border. Default=True.

    return_offset : :obj:`bool`, optional
        Specifies whether to return a tuple of the removed padding.
        Default=False.

    Returns
    -------
    Niimg-like object or :obj:`tuple`
        Cropped version of the input image and, if `return_offset=True`,
        a tuple of tuples representing the number of voxels
        removed (before, after) the cropped volumes, i.e.:
        *[(x1_pre, x1_post), (x2_pre, x2_post), ..., (xN_pre, xN_post)]*

    """
    try:
        data = load_img(img).get_fdata()
    except Exception as e:
        print(e)
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(
        data < -rtol * infinity_norm, data > rtol * infinity_norm
    )

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))

    # Sets full range if no data are found along the axis
    if coords.shape[1] == 0:
        start, end = np.array([0, 0, 0]), np.array(data.shape)
    else:
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    if pad:
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)][:3]
    cropped_im = _crop_img_to(img, slices, copy=copy)
    if mask:
        cropped_mask = _crop_img_to(mask, slices, copy=copy)
        return cropped_im, cropped_mask
    else:
        return cropped_im


def exclude_abnomral_slices(image, mask, slice_dim=-1):
    no_abnormal_image = []
    mask_slices = []
    if slice_dim == -1:
        for i in range(image.shape[slice_dim]):
            if (mask[..., i] > 0).mean() < .001:
                no_abnormal_image.append(image[..., i])
                mask_slices.append(mask[..., i])
    else:
        raise NotImplementedError(f'slice_dim = {slice_dim} is not supported')
    return torch.tensor(no_abnormal_image).permute((1, 2, 0)), torch.tensor(mask_slices).permute((1, 2, 0))


def TrainBrats(images_path: str, cfg, preload=True):
    # Assuming images and masks have the same naming convention and are in the same order

    split_file = f'split_{cfg.split_num}.pkl'  # Update with the path to the split file you want to load
    if os.path.exists(split_file):
        split_info = load_splits(images_path + '/' + split_file)
        all_images = split_info['train']
    else:
        all_images = sorted([f for f in os.listdir(images_path) if f.endswith('.nii.gz') and f.find('seg') == -1])
    val_files = all_images[-int(.1 * len(all_images)):]
    train_files = all_images[:-int(.1 * len(all_images))]
    # Get a list of corresponding mask files
    mask_train_files = sorted([f.replace('t1', 'seg') for f in train_files])
    mask_val_files = sorted([f.replace('t1', 'seg') for f in val_files])

    def get_files(image_files, mask_files):
        subjects = []
        counter = 0
        for img_file, mask_file in zip(image_files, mask_files):
            sub = nib.squeeze_image(nib.load(os.path.join(images_path, img_file)))
            if cfg.train_contains_tumor:
                mask = nib.squeeze_image(nib.load(os.path.join(images_path, mask_file)))
                image, mask = crop(sub, mask)
                mask = mask.get_fdata()
                image = image.get_fdata()
                image, mask = exclude_abnomral_slices(image, mask)
                image, mask = exclude_empty_slices(image, mask)
            else:
                image = crop(sub)
                image = image.get_fdata()
                image = exclude_empty_slices(image)

            # Call the preprocessing method
            image = image[None, ...]
            image = tio.ScalarImage(tensor=image)
            image = tio.Resize((240, 240, image.shape[-1]))(image)
            image = tio.CropOrPad((240, 240, image.shape[-1]))(image)
            brain_mask = (image.data > .001)
            subject_dict = {'vol': image, 'age': 70, 'ID': img_file, 'label': counter,
                            'Dataset': 'dummy', 'stage': 'stage', 'path': img_file,
                            'mask': tio.LabelMap(tensor=brain_mask)}
            subject = tio.Subject(subject_dict)
            subjects.append(subject)
            counter += 1
        if preload:
            manager = Manager()
            cache = DatasetCache(manager)
            ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
            ds = preload_wrapper(ds, cache, augment=get_augment(cfg))
        else:
            ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))
        if cfg.spatialDims == '2D':
            slice_ind = cfg.get('startslice', None)
            seq_slices = cfg.get('sequentialslices', None)
            ds = vol2slice(ds, cfg, slice=slice_ind, seq_slices=seq_slices)
        return ds

    train_set = get_files(train_files, mask_train_files)
    val_set = get_files(val_files, mask_val_files)
    return train_set, val_set


def EvalBrats(images_path: str, cfg):
    split_file = f'split_benchmark_{cfg.split_num}.pkl'  # Update with the path to the split file you want to load
    split_info = load_splits(images_path + '/' + split_file)
    test_files = split_info['test']
    val_files = split_info['val']
    # Get a list of corresponding mask files
    mask_test_files = sorted([f.replace('t1', 'seg') for f in test_files])
    mask_val_files = sorted([f.replace('t1', 'seg') for f in val_files])

    def get_files(image_files, mask_files):
        counter = 0
        subjects = []
        for img_file, mask_file in zip(image_files, mask_files):
            # Read MRI images using tio
            sub = nib.squeeze_image(nib.load(os.path.join(images_path, img_file)))
            mask = nib.squeeze_image(nib.load(os.path.join(images_path, mask_file)))
            image, mask = crop(sub, mask)
            image, mask = sub.get_fdata(), mask.get_fdata()
            image, mask = exclude_empty_slices(image, mask)
            image = image[None, ...]
            mask = mask[None, ...]
            brain_mask = (image > .001)
            subject_dict = {'vol': tio.ScalarImage(tensor=image), 'vol_orig': tio.ScalarImage(tensor=image),
                            'age': 70, 'ID': img_file, 'label': counter,
                            'Dataset': 'dataset', 'stage': 'dummy', 'path': img_file,
                            'mask': tio.LabelMap(tensor=brain_mask),
                            'mask_orig': tio.LabelMap(tensor=brain_mask),
                            'seg_available': True, 'seg': tio.LabelMap(tensor=mask),
                            'seg_orig': tio.LabelMap(tensor=mask)}
            subject = tio.Subject(subject_dict)
            subjects.append(subject)
            counter += 1
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        return ds

    test_set = get_files(test_files, mask_test_files)
    val_set = get_files(val_files, mask_val_files)
    return test_set, val_set
