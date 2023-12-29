from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import src.datamodules.create_dataset as create_dataset


class Brats21(LightningDataModule):

    def __init__(self, cfg, fold=None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.Brats21.IDs.val
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1', cfg.mode).str.replace(
                    'FLAIR.nii.gz', f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:  # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)


class Brats(LightningDataModule):
    def __init__(self, cfg, fold=None):
        super(Brats, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)

        self.cfg.permute = False  # no permutation for IXI
        self.brats_images_train = cfg.brats_images_train
        self.brats_images_val = cfg.brats_images_val
        self.brats_images_test_val = cfg.brats_images_test_val
        self.brats_images_test_test = cfg.brats_images_test_test
        states = ['train', 'val', 'test']

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'train'):
            if self.cfg.sample_set:  # for debugging
                raise NotImplementedError("this is not implemented yet.")
            else:
                # self.train = create_dataset.TrainBrats(self.brats_images_train, self.cfg)
                self.val = create_dataset.EvalBrats(self.brats_images_test_val, self.cfg)
                # self.val_eval = create_dataset.EvalBrats(self.brats_images_test_val, self.cfg)
                self.test_eval = create_dataset.EvalBrats(self.brats_images_test_test, self.cfg)

    # def train_dataloader(self):
    #     return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True,
    #                       shuffle=True, drop_last=self.cfg.get('droplast', False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)

    # def val_eval_dataloader(self):
    #     return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)


class MSLUB(LightningDataModule):

    def __init__(self, cfg, fold=None):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload', True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.MSLUB.IDs.val
        self.csvpath_test = cfg.path.MSLUB.IDs.test
        self.csv = {}
        states = ['val', 'test']

        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MSLUB'

            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase + '/Data/' + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('uniso/t1',
                                                                                      f'uniso/{cfg.mode}').str.replace(
                    't1.nii.gz', f'{cfg.mode}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self, 'val_eval'):
            if self.cfg.sample_set:  # for debugging
                self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else:
                self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True,
                          shuffle=False)
