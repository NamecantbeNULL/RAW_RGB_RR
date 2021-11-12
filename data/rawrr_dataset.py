"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np


def pack(image, pattern='RGGB'):
    """Pack RGGB Bayer planes into a 4-channel image."""
    shape = image.shape
    image1 = np.zeros((shape[0] // 2, shape[1] // 2, 4))
    if pattern == 'RGGB':
        image1[..., 0] = image[0::2, 0::2]
        image1[..., 1] = image[0::2, 1::2]
        image1[..., 2] = image[1::2, 0::2]
        image1[..., 3] = image[1::2, 1::2]
    elif pattern == 'GRBG':
        image1[..., 1] = image[0::2, 0::2]
        image1[..., 0] = image[0::2, 1::2]
        image1[..., 3] = image[1::2, 0::2]
        image1[..., 2] = image[1::2, 1::2]
    elif pattern == 'GBRG':
        image1[..., 1] = image[0::2, 0::2]
        image1[..., 3] = image[0::2, 1::2]
        image1[..., 0] = image[1::2, 0::2]
        image1[..., 2] = image[1::2, 1::2]
    return image1


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, pattern='RGGB'):
        I_rgb, T_rgb, R_rgb, I_raw, T_raw, R_raw = sample['I_rgb'], sample['T_rgb'], sample['R_rgb'], sample['I_raw'], sample['T_raw'], sample['R_raw']

        I_raw_packed = pack(I_raw, pattern=pattern)
        T_raw_packed = pack(T_raw, pattern=pattern)
        R_raw_packed = pack(R_raw, pattern=pattern)

        h, w = I_rgb.shape[:2]
        min_a = min(h, w)
        new_h = (min_a * 7 // 10) // 2 * 2
        new_w = (min_a * 7 // 10) // 2 * 2

        top = (np.random.randint(0, h - new_h)) // 2 * 2
        left = (np.random.randint(0, w - new_w)) // 2 * 2

        I_rgb = I_rgb[top: top + new_h, left: left + new_w]
        T_rgb = T_rgb[top: top + new_h, left: left + new_w]
        R_rgb = R_rgb[top: top + new_h, left: left + new_w]
        I_raw_packed = I_raw_packed[(top // 2): (top // 2 + new_h // 2), (left // 2): (left // 2 + new_w // 2)]
        T_raw_packed = T_raw_packed[(top // 2): (top // 2 + new_h // 2), (left // 2): (left // 2 + new_w // 2)]
        R_raw_packed = R_raw_packed[(top // 2): (top // 2 + new_h // 2), (left // 2): (left // 2 + new_w // 2)]

        return {'I_rgb': I_rgb, 'T_rgb': T_rgb, 'R_rgb': R_rgb, 'I_raw': I_raw_packed, 'T_raw': T_raw_packed, 'R_raw': R_raw_packed}


class RAWRRDataset(BaseDataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        if opt.phase == 'train':
            self.real_dir_rgbA1 = os.path.join(opt.dataroot, 'real_' + 'T_rgb')
            self.real_dir_rawA1 = os.path.join(opt.dataroot, 'real_' + 'T_raw')
            self.real_dir_rgbA2 = os.path.join(opt.dataroot, 'real_' + 'R_rgb')
            self.real_dir_rawA2 = os.path.join(opt.dataroot, 'real_' + 'R_raw')
            self.real_dir_rgbB = os.path.join(opt.dataroot, 'real_' + 'I_rgb')
            self.real_dir_rawB = os.path.join(opt.dataroot, 'real_' + 'I_raw')

            self.real_paths_rgbA1 = sorted(make_dataset(self.real_dir_rgbA1, opt.max_dataset_size))  # load images from '/path/to/data/real_T_rgb'
            self.real_paths_rawA1 = sorted(make_dataset(self.real_dir_rawA1, opt.max_dataset_size))  # load images from '/path/to/data/real_T_raw'
            self.real_paths_rgbA2 = sorted(
                make_dataset(self.real_dir_rgbA2, opt.max_dataset_size))  # load images from '/path/to/data/real_R_rgb'
            self.real_paths_rawA2 = sorted(
                make_dataset(self.real_dir_rawA2, opt.max_dataset_size))  # load images from '/path/to/data/real_R_raw'
            self.real_paths_rgbB = sorted(make_dataset(self.real_dir_rgbB, opt.max_dataset_size))  # load images from '/path/to/data/real_I_rgb
            self.real_paths_rawB = sorted(make_dataset(self.real_dir_rawB, opt.max_dataset_size))  # load images from '/path/to/data/real_I_raw'
            self.real_size = len(self.real_paths_rgbA1)  # get the size of real dataset

        self.crop = RandomCrop(opt.load_size)
        self.dir_rgbA1 = os.path.join(opt.dataroot, opt.phase + 'A1_rgb')
        self.dir_rawA1 = os.path.join(opt.dataroot, opt.phase + 'A1_raw')
        self.dir_rgbA2 = os.path.join(opt.dataroot, opt.phase + 'A2_rgb')
        self.dir_rawA2 = os.path.join(opt.dataroot, opt.phase + 'A2_raw')
        self.dir_rgbB = os.path.join(opt.dataroot, opt.phase + 'B_rgb')
        self.dir_rawB = os.path.join(opt.dataroot, opt.phase + 'B_raw')

        self.rgbA1_paths = sorted(make_dataset(self.dir_rgbA1, opt.max_dataset_size))  # load images from '/path/to/data/trainA1_rgb'
        self.rgbA2_paths = sorted(make_dataset(self.dir_rgbA2, opt.max_dataset_size)) # load images from '/path/to/data/trainA2_rgb'
        if not opt.phase == 'train':
            self.rawA1_paths = sorted(make_dataset(self.dir_rawA1, opt.max_dataset_size))  # load images from '/path/to/data/trainA1_raw'
            self.rawA2_paths = sorted(make_dataset(self.dir_rawA2, opt.max_dataset_size))  # load images from '/path/to/data/trainA2_raw'
            self.rgbB_paths = sorted(make_dataset(self.dir_rgbB, opt.max_dataset_size))  # load images from '/path/to/data/trainB_rgb'
            self.rawB_paths = sorted(make_dataset(self.dir_rawB, opt.max_dataset_size))
            self.B_size = len(self.rgbB_paths)  # get the size of dataset B

        self.A1_size = len(self.rgbA1_paths)  # get the size of dataset A1
        self.A2_size = len(self.rgbA2_paths)  # get the size of dataset A2

        input_nc = self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.output_nc  # get the number of channels of output image
        self.transform_A = transforms.Compose([transforms.ToTensor()])
        self.transform_B = transforms.Compose([transforms.ToTensor()])
        print(self.transform_A)

        # self.trans2 = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
        # self.trans4 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        is_real = random.random() <= 0.3
        if self.opt.phase == 'train':
            if is_real:
                real_index = index % self.real_size
                rgbA1_path = self.real_paths_rgbA1[real_index]  # make sure index is within then range
                rgbA2_path = self.real_paths_rgbA2[real_index]
                rgbB_path = self.real_paths_rgbB[real_index]
                rawA1_path = self.real_paths_rawA1[real_index]
                rawA2_path = self.real_paths_rawA2[real_index]
                rawB_path = self.real_paths_rawB[real_index]

                rgbA1_img = np.asarray(Image.open(rgbA1_path).convert('RGB'))
                rgbA2_img = np.asarray(Image.open(rgbA2_path).convert('RGB'))
                rgbB_img = np.asarray(Image.open(rgbB_path).convert('RGB'))
                rawA1_img = cv2.imread(rawA1_path, -1)
                rawA2_img = cv2.imread(rawA2_path, -1)
                rawB_img = cv2.imread(rawB_path, -1)

                if 'huawei' in rgbA1_path:
                    imgs = self.crop({'I_rgb': rgbB_img, 'T_rgb': rgbA1_img, 'R_rgb': rgbA2_img, 'I_raw': rawB_img,
                                      'T_raw':rawA1_img, 'R_raw': rawA2_img}, pattern='GRBG')
                else:
                    imgs = self.crop({'I_rgb': rgbB_img, 'T_rgb': rgbA1_img, 'R_rgb': rgbA2_img, 'I_raw': rawB_img,
                                      'T_raw': rawA1_img, 'R_raw': rawA2_img}, pattern='RGGB')
                rgbB_img, rgbA1_img, rgbA2_img, rawB_img, rawA1_img, rawA2_img = imgs['I_rgb'], imgs['T_rgb'], imgs['R_rgb'], imgs['I_raw'], imgs['T_raw'], imgs['R_raw']
                # A1_img, B_img = Image.fromarray(imgs['T']), Image.fromarray(imgs['I'])
                is_real_int = 1
            else:
                rgbA1_path = self.rgbA1_paths[index % self.A1_size]  # make sure index is within then range
                index_rgbA2 = random.randint(0, self.A2_size - 1)
                rgbA2_path = self.rgbA2_paths[index_rgbA2]
                rgbB_path = ''

                rgbA1_img = np.asarray(Image.open(rgbA1_path).convert('RGB'))
                rgbA2_img = np.asarray(Image.open(rgbA2_path).convert('RGB'))
                rgbB_img = np.zeros_like(rgbA1_img)
                rawA1_img = np.zeros((rgbA1_img.shape[0], rgbA1_img.shape[1])) + 400
                rawA2_img = np.zeros_like(rawA1_img) + 400
                rawB_img = np.zeros_like(rawA1_img) + 400
                rawA1_img = pack(rawA1_img, pattern='RGGB')
                rawA2_img = pack(rawA2_img, pattern='RGGB')
                rawB_img = pack(rawB_img, pattern='RGGB')

                is_real_int = 0
        else:
            rgbA1_path = self.rgbA1_paths[index]
            rawA1_path = self.rawA1_paths[index]
            rgbB_path = self.rgbB_paths[index]
            rawB_path = self.rawB_paths[index]
            rgbA1_img = np.asarray(Image.open(rgbA1_path).convert('RGB'))
            rgbA2_img = np.zeros_like(rgbA1_img)
            rgbB_img = np.asarray(Image.open(rgbB_path).convert('RGB'))
            rawA1_img = cv2.imread(rawA1_path, -1)
            rawA2_img = np.zeros_like(rawA1_img) + 400
            rawB_img = cv2.imread(rawB_path, -1)

            if 'huawei' in rgbA1_path:
                rawA1_img = pack(rawA1_img, pattern='GRBG')
                rawA2_img = pack(rawA2_img, pattern='GRBG')
                rawB_img = pack(rawB_img, pattern='GRBG')
            else:
                rawA1_img = pack(rawA1_img, pattern='RGGB')
                rawA2_img = pack(rawA2_img, pattern='RGGB')
                rawB_img = pack(rawB_img, pattern='RGGB')
            is_real_int = 1

        h, w, c = rgbA1_img.shape
        if self.opt.phase == 'test':
            if h > w:
                newh = 320
                neww = np.int(w / np.float(h) * 320)
                neww = neww // 32 * 32
            else:
                neww = 320
                newh = np.int(h / np.float(w) * 320)
                newh = newh // 32 * 32
            newh = newh * 2
            neww = neww * 2
            # resize = transforms.Resize([2 * newh, 2 * neww])
        else:
            newh = 256
            neww = 256
            # resize = transforms.Resize([256, 256])
        rgbA1_img = cv2.resize(rgbA1_img, (neww, newh))
        rgbA2_img = cv2.resize(rgbA2_img, (neww, newh))
        rgbB_img = cv2.resize(rgbB_img, (neww, newh))
        rawA1_img = cv2.resize(rawA1_img, (neww // 2, newh // 2))
        rawA2_img = cv2.resize(rawA2_img, (neww // 2, newh // 2))
        rawB_img = cv2.resize(rawB_img, (neww // 2, newh // 2))

        rgbA1 = self.transform_A(rgbA1_img)
        rgbA2 = self.transform_A(rgbA2_img)
        rgbB = self.transform_B(rgbB_img)

        if 'huawei' in rgbA1_path:
            rawA1_img = (rawA1_img - 256) / 4095
            rawA2_img = (rawA2_img - 256) / 4095
            rawB_img = (rawB_img - 256) / 4095
        else:
            rawA1_img = (rawA1_img - 400) / 15983
            rawA2_img = (rawA2_img - 400) / 15983
            rawB_img = (rawB_img - 400) / 15983
        rawA1 = torch.from_numpy(np.transpose(rawA1_img, (2, 0, 1)))
        rawA2 = torch.from_numpy(np.transpose(rawA2_img, (2, 0, 1)))
        rawB = torch.from_numpy(np.transpose(rawB_img, (2, 0, 1)))

        return {'T_rgb': rgbA1, 'R_rgb': rgbA2, 'I_rgb': rgbB, 'T_raw': rawA1, 'R_raw': rawA2, 'I_raw': rawB, 'B_paths': rgbB_path, 'isNatural': is_real_int}

    def __len__(self):
        """Return the total number of images."""
        if self.opt.dataset_size == 0 or self.opt.phase == 'test':
            length = max(self.A1_size, self.A2_size, self.B_size)
        else:
            length = self.opt.dataset_size
        return length


class TestDataset(BaseDataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.crop = RandomCrop(opt.load_size)
        self.dir_rgbA1 = os.path.join(opt.dataroot, 'testA1_rgb')
        self.dir_rawA1 = os.path.join(opt.dataroot, 'testA1_raw')
        self.dir_rgbA2 = os.path.join(opt.dataroot, 'testA2_rgb')
        self.dir_rawA2 = os.path.join(opt.dataroot, 'testA2_raw')
        self.dir_rgbB = os.path.join(opt.dataroot, 'testB_rgb')
        self.dir_rawB = os.path.join(opt.dataroot, 'testB_raw')


        self.rgbA1_paths = sorted(make_dataset(self.dir_rgbA1, opt.max_dataset_size))  # load images from '/path/to/data/testA1_rgb'
        self.rgbA2_paths = sorted(
            make_dataset(self.dir_rgbA2, opt.max_dataset_size))  # load images from '/path/to/data/testA2_rgb'
        self.rgbB_paths = sorted(make_dataset(self.dir_rgbB, opt.max_dataset_size))  # load images from '/path/to/data/testB_rgb'
        self.rawA1_paths = sorted(make_dataset(self.dir_rawA1, opt.max_dataset_size))
        self.rawA2_paths = sorted(make_dataset(self.dir_rawA2, opt.max_dataset_size))
        self.rawB_paths = sorted(make_dataset(self.dir_rawB, opt.max_dataset_size))
        self.B_size = len(self.rgbB_paths)  # get the size of dataset B
        self.A1_size = len(self.rgbA1_paths)  # get the size of dataset A1

        input_nc = self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.output_nc  # get the number of channels of output image
        self.transform_A = transforms.Compose([transforms.ToTensor()])
        self.transform_B = transforms.Compose([transforms.ToTensor()])
        print(self.transform_A)

        # self.trans2 = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
        # self.trans4 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        rgbA1_path = self.rgbA1_paths[index]
        rawA1_path = self.rawA1_paths[index]
        rgbA2_path = self.rgbA2_paths[index]
        rawA2_path = self.rawA2_paths[index]
        rgbB_path = self.rgbB_paths[index]
        rawB_path = self.rawB_paths[index]
        rgbA1_img = np.asarray(Image.open(rgbA1_path).convert('RGB'))
        rawA1_img = cv2.imread(rawA1_path, -1)
        rgbA2_img = np.asarray(Image.open(rgbA2_path).convert('RGB'))
        rawA2_img = cv2.imread(rawA2_path, -1)
        rgbB_img = np.asarray(Image.open(rgbB_path).convert('RGB'))
        rawB_img = cv2.imread(rawB_path, -1)

        if 'huawei' in rgbA1_path:
            rawA1_img = pack(rawA1_img, pattern='GRBG')
            rawA2_img = pack(rawA2_img, pattern='GRBG')
            rawB_img = pack(rawB_img, pattern='GRBG')
        else:
            rawA1_img = pack(rawA1_img, pattern='RGGB')
            rawA2_img = pack(rawA2_img, pattern='RGGB')
            rawB_img = pack(rawB_img, pattern='RGGB')
        is_real_int = 1
        is_real_int = 1

        h, w, c = rgbA1_img.shape
        if h > w:
            newh = 320
            neww = np.int(w / np.float(h) * 320)
            neww = neww // 32 * 32
        else:
            neww = 320
            newh = np.int(h / np.float(w) * 320)
            newh = newh // 32 * 32
        rgbA1_img = cv2.resize(rgbA1_img, (neww * 2, newh * 2))
        rgbA2_img = cv2.resize(rgbA2_img, (neww * 2, newh * 2))
        rgbB_img = cv2.resize(rgbB_img, (neww * 2, newh * 2))
        rawA1_img = cv2.resize(rawA1_img, (neww, newh))
        rawA2_img = cv2.resize(rawA2_img, (neww, newh))
        rawB_img = cv2.resize(rawB_img, (neww, newh))

        rgbA1 = self.transform_A(rgbA1_img)
        rgbA2 = self.transform_A(rgbA2_img)
        rgbB = self.transform_B(rgbB_img)

        if 'huawei' in rgbA1_path:
            rawA1_img = (rawA1_img - 256) / 4095
            rawA2_img = (rawA2_img - 256) / 4095
            rawB_img = (rawB_img - 256) / 4095
        else:
            rawA1_img = (rawA1_img - 400) / 15983
            rawA2_img = (rawA2_img - 400) / 15983
            rawB_img = (rawB_img - 400) / 15983

        rawA1 = torch.from_numpy(np.transpose(rawA1_img, (2, 0, 1)))
        rawA2 = torch.from_numpy(np.transpose(rawA2_img, (2, 0, 1)))
        rawB = torch.from_numpy(np.transpose(rawB_img, (2, 0, 1)))

        # T2 = self.trans2(A1_img)
        # T4 = self.trans4(A1_img)
        return {'T_rgb': rgbA1, 'R_rgb': rgbA2, 'I_rgb': rgbB, 'T_raw': rawA1, 'R_raw': rawA2, 'I_raw': rawB, 'B_paths': rgbB_path, 'isNatural': is_real_int}

    def __len__(self):
        """Return the total number of images."""
        length = max(self.A1_size, self.B_size)
        return length


class TestDataset_wogt(BaseDataset):
    """A reflection dataset class to load data from A1, A2, B datasets, where A1(indoor) and A2(outdoor) are image sets
     without reflection, and B is a image set with reflection."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.crop = RandomCrop(opt.load_size)
        self.dir_rgbB = os.path.join(opt.dataroot, 'wogtB_rgb')
        self.dir_rawB = os.path.join(opt.dataroot, 'wogtB_raw')

        self.rgbB_paths = sorted(make_dataset(self.dir_rgbB, opt.max_dataset_size))  # load images from '/path/to/data/testB_rgb'
        self.rawB_paths = sorted(make_dataset(self.dir_rawB, opt.max_dataset_size))
        self.B_size = len(self.rgbB_paths)  # get the size of dataset B

        input_nc = self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.output_nc  # get the number of channels of output image
        self.transform_B = transforms.Compose([transforms.ToTensor()])
        print(self.transform_B)

        # self.trans2 = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
        # self.trans4 = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        rgbB_path = self.rgbB_paths[index]
        rawB_path = self.rawB_paths[index]

        rgbB_img = np.asarray(Image.open(rgbB_path).convert('RGB'))
        rawB_img = cv2.imread(rawB_path, -1)
        rgbA1_img = np.zeros_like(rgbB_img)
        rawA1_img = np.zeros_like(rawB_img)
        rgbA2_img = np.zeros_like(rgbB_img)
        rawA2_img = np.zeros_like(rawB_img)

        if 'D90' in rgbB_path:
            rawA1_img = pack(rawA1_img, pattern='GBRG')
            rawA2_img = pack(rawA2_img, pattern='GBRG')
            rawB_img = pack(rawB_img, pattern='GBRG')
        else:
            rawA1_img = pack(rawA1_img, pattern='RGGB')
            rawA2_img = pack(rawA2_img, pattern='RGGB')
            rawB_img = pack(rawB_img, pattern='RGGB')
        is_real_int = 1

        h, w, c = rgbA1_img.shape
        if h > w:
            newh = 320
            neww = np.int(w / np.float(h) * 320)
            neww = neww // 32 * 32
        else:
            neww = 320
            newh = np.int(h / np.float(w) * 320)
            newh = newh // 32 * 32
        rgbA1_img = cv2.resize(rgbA1_img, (neww * 2, newh * 2))
        rgbA2_img = cv2.resize(rgbA2_img, (neww * 2, newh * 2))
        rgbB_img = cv2.resize(rgbB_img, (neww * 2, newh * 2))
        rawA1_img = cv2.resize(rawA1_img, (neww, newh))
        rawA2_img = cv2.resize(rawA2_img, (neww, newh))
        rawB_img = cv2.resize(rawB_img, (neww, newh))

        rgbA1 = self.transform_B(rgbA1_img)
        rgbA2 = self.transform_B(rgbA2_img)
        rgbB = self.transform_B(rgbB_img)

        if 'D90' in rgbB_path:
            rawB_img = (rawB_img - 0) / 4095
        elif 'D7500' in rgbB_path:
            rawB_img = (rawB_img - 400) / 15983
        elif 'Honor' in rgbB_path:
            rawB_img = (rawB_img - 64) / 959

        rawA1 = torch.from_numpy(np.transpose(rawA1_img, (2, 0, 1)))
        rawA2 = torch.from_numpy(np.transpose(rawA2_img, (2, 0, 1)))
        rawB = torch.from_numpy(np.transpose(rawB_img, (2, 0, 1)))

        # T2 = self.trans2(A1_img)
        # T4 = self.trans4(A1_img)
        return {'T_rgb': rgbA1, 'R_rgb': rgbA2, 'I_rgb': rgbB, 'T_raw': rawA1, 'R_raw': rawA2, 'I_raw': rawB, 'B_paths': rgbB_path, 'isNatural': is_real_int}

    def __len__(self):
        """Return the total number of images."""
        length = self.B_size
        return length