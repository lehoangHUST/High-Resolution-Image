import cv2
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


class SRDataset(Dataset):
    def __init__(self, path: str, task: str = 'train'):
        super(SRDataset, self).__init__()
        self.task = task
        self.path = path
        self.path_HR = os.path.join(self.path, self.task, 'HR')
        self.path_LR = os.path.join(self.path, self.task, 'LR')

    def get_csv(self):
        csv_path = os.path.join(self.path, f"{self.task}.csv")
        try:
            dataframe = pd.read_csv(csv_path)
        except FileNotFoundError:
            column_name = ["filename HighResolution", "filename LowResolution"]
            value_list = []
            for i in range(len(self)):
                imgname_HR, imgname_LR = os.listdir(self.path_LR)[i], os.listdir(self.path_HR)[i]
                value = (os.path.join(self.path_HR, imgname_HR),
                         os.path.join(self.path_LR, imgname_LR))
                value_list.append(value)
            dataframe = pd.DataFrame(value_list, columns=column_name)
            dataframe.to_csv(csv_path, index=None)
            print('Successfully created the CSV file: {}'.format(csv_path))
            dataframe = pd.read_csv(csv_path)
        return dataframe

    def __getitem__(self, idx):
        dataframe = self.get_csv()
        imgname_HR, imgname_LR = dataframe.iloc[idx, 0], dataframe.iloc[idx, 1]
        imgHR, imgLR = cv2.imread(imgname_HR), cv2.imread(imgname_LR)
        # Convert
        imgHR, imgLR = imgHR[:, :, ::-1], imgLR[:, :, ::-1]
        imgHR, imgLR = imgHR.transpose(2, 0, 1), imgLR.transpose(2, 0, 1)  # (C,H,W)
        imgHR, imgLR = np.ascontiguousarray(imgHR) / 255, np.ascontiguousarray(imgLR) / 255
        sample = {'LR_imagename': imgname_LR,
                  'HR_imagename': imgname_HR,
                  'LR_img': torch.from_numpy(imgLR),
                  'HR_img': torch.from_numpy(imgHR)}
        sample["LR_img"] = transforms.Resize((384, 384))(sample["LR_img"])
        return sample

    def __len__(self):
        return len(os.listdir(self.path_LR))

