# -*- coding:utf-8 -*-
# Coder: Yao Zhao
# Github: https://github.com/yyyzzzhao
# creat my dataset with mask
# ==============================================================================

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class MaskDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B,M}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABM = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABM_paths = sorted(make_dataset(self.dir_ABM, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, M, A_paths, B_paths, M_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            M (tensor) - - 
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            M_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABM_path = self.ABM_paths[index]
        ABM = Image.open(ABM_path).convert('RGB')
        # split AB image into A and B
        w, h = ABM.size
        w2 = int(w / 3)
        A = ABM.crop((0, 0, w2, h))  #  left, upper, right, lower
        B = ABM.crop((w2, 0, 2*w2, h))
        M = ABM.crop((2*w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        M_transform = get_transform(self.opt, transform_params, grayscale=True)

        A = A_transform(A)
        B = B_transform(B)
        M = M_transform(M)

        return {'A': A, 'B': B, 'M': M, 'A_paths': ABM_path, 'B_paths': ABM_path, 'M_paths': ABM_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABM_paths)