"""
semantic segmentation dataset
"""
import torch
import os
import pandas as pd
import numpy as np
import cv2


class BD100KSegdataset_one_hot(torch.utils.data.Dataset):

    def __init__(self, csv_file, img_dir, label_dir, transform = None , num_classes = 20):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_dir = label_dir
        self.num_classes  = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        mask = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        img = cv2.imread(img)
        mask = cv2.imread(mask,0)
        mask[mask==255] = 19

        if self.transform:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        mask = np.asarray(mask)
        (h, w) = mask.shape
        label = np.zeros((self.num_classes,h,w))
        class_ids = np.unique(mask)
        for class_id in class_ids:
            label[int(class_id)] = np.where(mask == int(class_id), 1, label[int(class_id)])

        return img, torch.Tensor(label).squeeze(0)





















