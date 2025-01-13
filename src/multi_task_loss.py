"""
Multi task loss for semantic segmentation,lane_marking,drivable_area and object_detection multi-task network
"""
import torch
import numpy as np
import torch.nn as nn
from utils import intersection_over_union


# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


class Multi_task_loss_fn(nn.Module):

    def __init__(self, tasks=None, scaled_anchors=None, DEVICE="cuda", WEIGHTED=False):
        super(Multi_task_loss_fn, self).__init__()
        self.tasks = tasks  # same as tasks mentioned in train.py file
        self.scaled_anchors = scaled_anchors  # predefined anchors in train.py
        self.device = DEVICE  # "cuda" or "cpu"
        self.weighted = WEIGHTED  # If True calculates weights for each task
        self.mse = nn.MSELoss()
        for task in self.tasks:
            if task[0] == "semantic_segmentation":
                self.seg_loss = WCE_L(num_classes=task[1], DEVICE=self.device)  # Initializing the segmentation loss
            elif task[0] == "lane_marking":
                self.lane_loss = WCE_L(num_classes=task[1], DEVICE=self.device)
            elif task[0] == "drivable_area":
                self.drivable_loss = WCE_L(num_classes=task[1], DEVICE=self.device)
            if task[0] == "object_detection":
                self.yolo_loss = YoloLoss()  # Initializing YOLOv3 loss

    def forward(self, preds, targets, features):

        features_avg = torch.zeros_like(features[self.tasks[0][0]])
        for task in self.tasks:
            features_avg += features[task[0]]
        features_avg = features_avg / len(features)
        losses = torch.zeros(len(self.tasks)).to(self.device)
        feature_losses = torch.zeros(len(self.tasks)).to(self.device)
        for i, task in enumerate(self.tasks):
            if task[0] == "semantic_segmentation":
                seg_loss = self.seg_loss(preds[task[0]], targets[task[0]].to(self.device, dtype=torch.long))
                losses[i] = seg_loss
                feature_losses[i] = self.mse(features[task[0]], features_avg)
            elif task[0] == "lane_marking":
                lane_loss = self.lane_loss(preds[task[0]].to(self.device),
                                           targets[task[0]].to(self.device, dtype=torch.long))
                losses[i] = lane_loss
                feature_losses[i] = self.mse(features[task[0]], features_avg)

            elif task[0] == "drivable_area":
                drivable_loss = self.drivable_loss(preds[task[0]].to(self.device),
                                                   targets[task[0]].to(self.device, dtype=torch.long))
                losses[i] = drivable_loss
                feature_losses[i] = self.mse(features[task[0]], features_avg)

            elif task[0] == "object_detection":
                yolo_loss = (
                        self.yolo_loss(preds[task[0]][0], targets[task[0]][0].to(self.device), self.scaled_anchors[0])
                        + self.yolo_loss(preds[task[0]][1], targets[task[0]][1].to(self.device), self.scaled_anchors[1])
                        + self.yolo_loss(preds[task[0]][2], targets[task[0]][2].to(self.device), self.scaled_anchors[2])
                )
                losses[i] = yolo_loss
                feature_losses[i] = self.mse(features[task[0]], features_avg)

        # weights = torch.FloatTensor([(sum(losses)/(len(self.tasks)*losses[i])) for i in range(len(self.tasks))]).to(DEVICE)

        if self.weighted:
            loss_avg = torch.mean(losses)
            weights = torch.FloatTensor([1 + (loss_avg / losses[i]) for i in range(len(self.tasks))]).to(self.device)
            loss = losses * weights
        else:
            loss = losses

        loss = sum(loss) + sum(feature_losses)

        task_losses = losses.detach().cpu()

        return loss, task_losses, sum(feature_losses)


class WCE_L(nn.Module):

    def __init__(self, num_classes=20, DEVICE="cuda"):
        super(WCE_L, self).__init__()
        self.num_classes = num_classes
        self.device = DEVICE

    def forward(self, preds, targets, smooth=1):

        weights = np.ones(self.num_classes, dtype=float)
        class_id_counts = np.ones(self.num_classes, dtype=float)
        labels_array = np.asarray(targets.cpu())
        class_ids, count = np.unique(labels_array, return_counts=True)
        length_of_data = labels_array.size
        for idx, class_id in enumerate(class_ids):
            class_id_counts[int(class_id)] += count[idx]
        for idx, class_id_count in enumerate(class_id_counts):
            weights[idx] += length_of_data + smooth / ((self.num_classes * class_id_count) + smooth)
        weights = torch.Tensor(weights).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(preds, targets)

        return loss


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1.5
        self.lambda_noobj = 1
        self.lambda_obj = 1.5
        self.lambda_box = 1.5

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
        )
