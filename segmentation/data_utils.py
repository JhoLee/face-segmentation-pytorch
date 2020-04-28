import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

class CelebAMaskDataset(Dataset):
    """
    CelebA Mask HQ Dataset
    https://github.com/switchablenorms/CelebAMask-HQ
    """
    
    def __init__(self, img_dir, mask_dir, csv_path, transform=None):
        """
        #todo: Add comment here.

        :param img_dir: Directory path to original images.
        :param mask_dir: Directory path to mask images.
        :param transform:
        :type img_dir: str
        :type mask_dir: str
        :type transform: callable
        """

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data_list = pd.read_csv(csv_path)
        self.transform = transform
    
    def __len__(self):
        """
        
        :return: length of images 
        """
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        name = self.data_list[idx]
        img = io.imread(os.path.join(img_dir, name))
        mask = io.imread(os.path.join(mask_dir, name))

        return name, img, mask
