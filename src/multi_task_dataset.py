"""
dataset loader for semantic segmentation,lane_marking,drivable_area and object_detection (yolov3)
"""

import pandas as pd
import numpy as np
import cv2
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
    transforms,
    plot_image_cv2,
)


class multi_task_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, dataset_dir, label_dir=None, transform=None, img_dir=None,
                 tasks=None, one_hot_label=False, anchors=None, S=None):

        self.annotations = pd.read_csv(csv_file)  # csv file in the form of image name, task labels in cosecutive colomns
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

        if self.one_hot_lable:
            for task in self.tasks:
                if task[0] != "object_detection" and labels[task] != "empty":
                    (h, w) = labels[task].shape
                    label = np.zeros((task[1], h, w))
                    class_ids = np.unique(labels[task])
                    for class_id in class_ids:
                        label[int(class_id)] = np.where(labels[task] == int(class_id), 1, label[int(class_id)])
                    label[task] = label

        if masks["object_detection"] is not None:

            for box in labels["object_detection"]:
                iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
                anchor_indices = iou_anchors.argsort(descending=True, dim=0)
                x, y, width, height, class_label = box
                if class_label > 9:
                    continue
                elif class_label == 3:
                    class_label = 0
                elif class_label == 8:
                    class_label = 2
                elif class_label == 9:
                    class_label = 3  # because class 3 is empty now. class 3 is changed to 0
                has_anchor = [False] * 3  # each scale should have one anchor

                for anchor_idx in anchor_indices:
                    scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')
                    anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                    S = self.S[scale_idx]
                    i, j = int(S * y), int(S * x)  # which cell
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                        width_cell, height_cell = (
                            width * S,
                            height * S,
                        )  # can be greater than 1 since it's relative to cell
                        box_coordinates = torch.tensor(
                            [x_cell, y_cell, width_cell, height_cell]
                        )
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True

                    elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

            labels["object_detection"] = targets

        return img, labels


def test():
    from utils import mask_to_colormap
    from torchvision.utils import save_image
    IMAGE_HEIGHT = 704
    IMAGE_WIDTH = 704
    train_transform, test_transform = transforms(IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT)
    S = [IMAGE_WIDTH // 32, IMAGE_WIDTH // 16, IMAGE_WIDTH // 8]
    anchors = [
        [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
        [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
        [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],
    ]
    scaled_anchors = torch.tensor(anchors) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    DATASET_DIR = "D:\Thesis\A2D2\Object_detection_bboxes/"
    dataset = multi_task_dataset(csv_file="train.csv",
                                 dataset_dir=DATASET_DIR,
                                 label_dir=["1.sem_seg_masks/", "2.lane_masks/", "3.drivable_masks/", "4.det_labels/"],
                                 # label_dir=["1.sem_seg_masks/","4.det_labels/"],
                                 transform=test_transform,
                                 tasks=[["semantic_segmentation", 20], ["lane_marking", 6],
                                        ["drivable_area", 3], ["object_detection", 8]],
                                 # tasks=[["semantic_segmentation", 20],["object_detection", 16]],
                                 anchors=anchors,
                                 img_dir="D:\Thesis\A2D2\Object_detection_bboxes\images",
                                 S=S, )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    img_id = 0
    for x, target in loader:
        # for key in target.keys():
        #    # print(y[key])
        #    if target[key] != ['empty'] and key != "object_detection":
        #        print(target[key].shape)
        #    elif len(target[key]) == 3:
        #        print(target[key][0].shape)
        #        print(target[key][1].shape)
        #        print(target[key][2].shape)
        boxes = []

        y = target["object_detection"]
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(x[0].shape)
        # save_image(x[0], "saved_with_tensor_cv2_after.png")
        im, original = plot_image_cv2(x[0], boxes)
        print(img_id, boxes)
        cv2.imwrite("det_pillow_after" + str(img_id) + ".png", im)
        img_id += 1
        # cv2.imwrite("det2.png", original)
        # save_image(x[0], "img.png")
        # sem_seg = mask_to_colormap(np.repeat((np.asarray(target["semantic_segmentation"].permute(1, 2, 0))), 3, 2),
        #                           task="semantic_segmentation")
        # cv2.imwrite("sem_seg.png", sem_seg)
        # lane = mask_to_colormap(np.repeat((np.asarray(target["lane_marking"].permute(1, 2, 0))), 3, 2),
        #                        task="lane_marking")
        # cv2.imwrite("lane.png", lane)
        # drivable = mask_to_colormap(np.repeat((np.asarray(target["drivable_area"].permute(1, 2, 0))), 3, 2),
        #                            task="drivable_area")
        # cv2.imwrite("drivable.png", drivable)


if __name__ == "__main__":
    test()
