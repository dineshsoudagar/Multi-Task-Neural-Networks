"""

calculates the inference time over some examples for single and multi task model

"""

import os
import torch
import torch.optim as optim
from multi_task_model import Multi_task_model
from multi_task_loss import Multi_task_loss_fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import (
    save_model,
    check_accuracies,
    get_loaders,
    transforms,
    save_some_examples,
    preds_to_class_map,
    mask_to_colormap,
    cells_to_bboxes,
    non_max_suppression,
    plot_image_cv2_for_overlay,
    segmentation_color_ids,
    lane_marking_color_ids,
    drivable_class_ids,
    combined_color_ids,
)
import cv2
from PIL import Image
import numpy as np
import glob

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
# TASKS = [["semantic_segmentation", 21], ["lane_marking", 5], ["drivable_area", 3], ["object_detection", 8]]
tasks_name = "semantic_segmentation"
BACKBONE = "resnet34"
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 704
CONF_THRESHOLD = 0.70  # confidence of yolo model for the prediction
IOU_THRESH = 0.5
NMS_THRESH = 0.3
SAVE_MODEL_FILE = "models/" + tasks_name + "/"
SAVE_PATH = "predictions/" + tasks_name + "/"
if not os.path.exists(SAVE_MODEL_FILE):
    os.makedirs(SAVE_MODEL_FILE)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
DATASET_DIR = "D:\Thesis\A2D2\Object_detection_bboxes/"
IMG_DIR = "D:/Thesis/A2D2/Object_detection_bboxes/images/"
# LABEL_DIR = ["1.sem_seg_masks/", "2.lane_masks/", "3.drivable_masks/", "4.det_labels/"]
LABEL_DIR = ["1.sem_seg_masks/"]

S = [IMAGE_WIDTH // 32, IMAGE_WIDTH // 16, IMAGE_WIDTH // 8]
ANCHORS = [[[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
           [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
           [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]]
scaled_anchors = (torch.tensor(ANCHORS) * (torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))).to(DEVICE)
train_transform, test_transform = transforms(IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT)
dummy_bbox = [[0.5, 0.5, 0.1, 0.1, 0]]


def post_processing(preds, task=None):
    if task[0] != "object_detection":
        preds = preds_to_class_map(preds)
        preds = np.repeat(np.asarray(preds[0]), 3, 2)  # (H,W,3)
        # preds = mask_to_colormap(preds, task=task[0])
        preds_resized = test_transform_after(image=img_, mask=preds, mask2=dummy_mask, mask3=dummy_mask,
                                             bboxes=dummy_bbox)
        final_mask = np.asarray(preds_resized["mask"])
        final_mask = final_mask.astype(np.uint8)

    else:
        bboxes = [[] for _ in range(1)]  # change range for different batch size
        yolo_out = preds
        for i in range(3):
            batch_size, A, S, _, _ = yolo_out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_bboxes(
                yolo_out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        nms_boxes = non_max_suppression(
            bboxes[0], iou_threshold=NMS_THRESH, threshold=CONF_THRESHOLD, box_format="midpoint",
        )
        final_mask = nms_boxes

    return final_mask


def combine_masks(masks=None, tasks=None):
    # combined_mask = np.zeros_like(masks[tasks[0][0]])
    for task in tasks:
        if task[0] == "lane_marking":
            class_ids = np.unique(masks[task[0]])
            for class_id in class_ids:
                if class_id != 4:
                    masks["drivable_area"] = np.where(
                        masks[task[0]] == [[class_id, class_id, class_id]],
                        [[class_id + 21, class_id + 21, class_id + 21]],
                        masks["drivable_area"])

            masks["drivable_area"][masks["drivable_area"] == 0] = 28
            masks["drivable_area"][masks["drivable_area"] == 1] = 27
            masks["drivable_area"][masks["drivable_area"] == 2] = 28

        elif task[0] == "drivable_area":
            class_ids = np.unique(masks[task[0]])
            for class_id in class_ids:
                if class_id != 28:
                    masks["semantic_segmentation"] = np.where(
                        masks[task[0]] == [[class_id, class_id, class_id]],
                        [[class_id, class_id, class_id]],
                        masks["semantic_segmentation"])
        else:
            raise ValueError(
                "invalid argument : task should be one of these values 'semantic_segmentation', 'lane_marking', "
                "'drivable_area'")

    combined_mask = mask_to_colormap(masks["semantic_segmentation"], task="combined")

    combined_mask = combined_mask.astype(np.uint8)

    return combined_mask


def get_prediction(model, input_img, task=None):
    with torch.no_grad():
        output = model(input_img)

    if task[0] != "object_detection":
        output_mask = post_processing(output[task[0]], task=task)
    else:
        output_mask = post_processing(output[task[0]], task=task)

    return output_mask


def get_overlay_output(img_original, mask, task=None):
    if task[0] != "object_detection":
        overlay_output = cv2.addWeighted(mask, 0.4, img_original, 1 - 0.6, 0)
    else:
        det = plot_image_cv2_for_overlay(img_original, mask)
        overlay_output = det
    return overlay_output


def load_model(tasks, name):
    print("---loading model---")
    model = Multi_task_model(backbone=BACKBONE, in_channels=3, tasks=tasks)
    model.to(DEVICE)
    name = "models/" + name + "/" + tasks[0][0] + ".pth"
    model.load_state_dict(torch.load(name, map_location=DEVICE))
    model.eval()

    return model


def load_multi_model(tasks, name):
    print("---loading model---")
    model = Multi_task_model(backbone=BACKBONE, in_channels=3, tasks=tasks)
    model.to(DEVICE)
    name = "models/" + name + "/" + name + ".pth"
    model.load_state_dict(torch.load(name, map_location=DEVICE))
    model.eval()

    return model


train_transform_after, test_transform_after = transforms(IMAGE_WIDTH=1920, IMAGE_HEIGHT=1208)

image_list = glob.glob(
    "D:\Thesis\A2D2\camera_lidar-20190401121727_camera_frontcenter\camera_lidar/20190401_121727\camera\cam_front_center_6to12sec/*.png")


def main():
    # TASKS = [["semantic_segmentation", 21], ["lane_marking", 5], ["drivable_area", 3], ["object_detection", 8]]
    TASKS = [["lane_marking", 5], ["drivable_area", 3], ["object_detection", 8]]
    # sem_model = load_model(tasks=[["semantic_segmentation", 21]], name="semantic_segmentation")
    # lane_model = load_model(tasks=[["lane_marking", 5]], name="lane")
    # dri_model = load_model(tasks=[["drivable_area", 3]], name="dri")
    # det_model = load_model(tasks=[["object_detection", 8]], name="det")
    torch.cuda.empty_cache()
    model = load_multi_model(tasks=TASKS, name="lane_dri_det_WL_FL")
    # model = dri_model
    warm_up_input = torch.rand(1, 3, 704, 704).to(DEVICE)
    with torch.no_grad():  # warmup
        for i in range(20):
            _ = model(warm_up_input)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = len(image_list)
    print("examples", repetitions)
    timings = np.zeros((repetitions, 1))
    for i, image in tqdm(enumerate(image_list)):
        input_img = np.array(Image.open(image))
        dummy_mask = np.ones_like(input_img)
        augmented = test_transform(image=input_img, mask=dummy_mask, mask2=dummy_mask, mask3=dummy_mask,
                                   bboxes=dummy_bbox)
        input_img = augmented["image"]
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(DEVICE)
        with torch.no_grad():
            starter.record()
            _ = model(input_img)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    mean_sync = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("mean time", mean_sync)


if __name__ == "__main__":
    main()
