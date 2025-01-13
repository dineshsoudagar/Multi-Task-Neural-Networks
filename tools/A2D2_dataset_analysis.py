"""
Complete analysis of A2D2 dataset
calculates number of instances per class for all task labels in A2D2 dataset.
"""

import glob
import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    transforms,
)


class multi_task_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, dataset_dir, label_dir=None, transform=None, img_dir=None,
                 tasks=None, one_hot_label=False, anchors=None, S=None):

        self.annotations = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.label_dir = label_dir
        self.tasks = tasks
        self.one_hot_lable = one_hot_label
        self.anchors = anchors
        self.img_dir = img_dir
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img = np.array(Image.open(img_path))
        labels = {}
        masks = {"semantic_segmentation": None, "lane_marking": None, "drivable_area": None, "object_detection": None}
        dummy_mask = np.ones_like(img)
        dummy_bbox = [[0.5, 0.5, 0.1, 0.1, 0]]

        for i, task in enumerate(self.tasks):

            if task[0] != "object_detection":
                mask = os.path.join(self.dataset_dir, self.label_dir[i], self.annotations.iloc[index, 1])
                mask = cv2.imread(mask, 0)
                mask[mask == 255] = task[1] - 1
                masks[task[0]] = mask

            elif task[0] == "object_detection":
                label_path = os.path.join(self.dataset_dir, self.label_dir[i], self.annotations.iloc[index, 2])
                bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
                masks[task[0]] = bboxes
                targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        if self.transform:
            augmentations = self.transform(
                image=img,
                mask=masks["semantic_segmentation"] if masks["semantic_segmentation"] is not None else dummy_mask,
                mask2=masks["lane_marking"] if masks["lane_marking"] is not None else dummy_mask,
                mask3=masks["drivable_area"] if masks["drivable_area"] is not None else dummy_mask,
                bboxes=masks["object_detection"] if masks["object_detection"] is not None else dummy_bbox
            )

            img = augmentations["image"]
            semantic_segmentation = augmentations["mask"]
            lane_marking = augmentations["mask2"]
            drivable_area = augmentations["mask3"]
            object_detection = augmentations["bboxes"]

            labels["semantic_segmentation"] = semantic_segmentation if masks[
                                                                           "semantic_segmentation"] is not None else "empty"
            labels["lane_marking"] = lane_marking if masks["lane_marking"] is not None else "empty"
            labels["drivable_area"] = drivable_area if masks["drivable_area"] is not None else "empty"
            labels["object_detection"] = object_detection if masks["object_detection"] is not None else "empty"

        return img, labels


IMAGE_HEIGHT = 704
IMAGE_WIDTH = 704
train_transform, test_transform = transforms(IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT)
S = [IMAGE_WIDTH // 32, IMAGE_WIDTH // 16, IMAGE_WIDTH // 8]
anchors = [
    [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
    [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
    [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],
]
TASKS = [["semantic_segmentation", 21], ["lane_marking", 5], ["drivable_area", 3], ["object_detection", 14]]
LABEL_DIR = ["1.sem_seg_masks/", "2.lane_masks/", "3.drivable_masks/",
             "4.det_labels/"]
IMAGE_DIR = "D:\Thesis\A2D2\Object_detection_bboxes\images"
DATASET_DIR = "D:\Thesis\A2D2\Object_detection_bboxes/"
scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
)

files_list = glob.glob("D:\Thesis/5.multi_task_model_A2D2_pillow\A2D2_csv_sets/*.csv")


def calculate_dataset_class_distribution():
    text_file = open("distribution_proper.txt", "w")
    total = {}
    distributions_percentage = {file.rsplit("\\")[-1]: {} for file in files_list}
    distributions = {file.rsplit("\\")[-1]: {} for file in files_list}
    for task in TASKS:
        total[task[0]] = {class_id: 0 for class_id in range(task[1])}
    for file in files_list:
        set_id = file.rsplit("\\")[-1]
        print(set_id)
        dataset = multi_task_dataset(csv_file=file,
                                     dataset_dir=DATASET_DIR,
                                     label_dir=LABEL_DIR,
                                     transform=test_transform,
                                     tasks=TASKS,
                                     anchors=anchors,
                                     img_dir=IMAGE_DIR,
                                     S=S, )
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for task in TASKS:
            distributions[set_id][task[0]] = {class_id: 0 for class_id in range(task[1])}

            distributions_percentage[set_id][task[0]] = {class_id: 0 for class_id in range(task[1])}

        for x, target in tqdm(loader):
            for task in TASKS:
                if task[0] != "object_detection":
                    for class_id in np.unique(target[task[0]]):
                        distributions[set_id][task[0]][class_id] += 1
                        total[task[0]][class_id] += 1
                elif task[0] == "object_detection":
                    for class_id in target[task[0]]:
                        distributions[set_id][task[0]][int(class_id[-1])] += 1
                        total[task[0]][int(class_id[-1])] += 1

                #print(total[task[0]])
                #print(distributions[set_id][task[0]])

        for task in TASKS:
            print(total[task[0]])
            print(distributions[set_id])

    for file in files_list:
        set_id = file.rsplit("\\")[-1]
        text_file.write(set_id)
        for distribution in distributions_percentage[set_id]:
            for key in distributions_percentage[set_id][distribution].keys():
                distributions_percentage[set_id][distribution][key] = str(round(
                    (distributions[set_id][distribution][key] / total[distribution][key]) * 100, 2)) + "%"
        for distribution in distributions[set_id]:
            print(set_id)
            print(distribution)
            print(total[distribution])
            print(distributions[set_id][distribution])
            print(distributions_percentage[set_id][distribution])
            # file.write("\n".join(distributions[distributions]))
            text_file.write("\n")
            text_file.write(str(distribution))
            text_file.write("\n")
            text_file.write(str(total[distribution]))
            text_file.write("\n")
            text_file.write(str(distributions[set_id][distribution]))
            text_file.write("\n")
            text_file.write(str(distributions_percentage[set_id][distribution]))
            text_file.write("\n")
    text_file.close()


if __name__ == "__main__":
    calculate_dataset_class_distribution()
