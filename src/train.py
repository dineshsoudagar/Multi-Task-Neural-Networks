"""
Training file for all single task and multi task networks

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
    load_model,
    check_accuracies,
    get_loaders,
    transforms,
    save_some_examples,
)

# Hyperparameters

"""
mention the tasks.
TASKS : list [["task name", no. of classes],["task name", no. of classes]....]
task names : "semantic_segmentation", "lane_marking", "drivable_area", "object_detection"

BACKBONE : one of resnet18, resnet34, resnet50, resnet101, resnet150. 

"""

LEARNING_RATE = 0.00001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
TASKS = [["semantic_segmentation", 21], ["lane_marking", 5], ["drivable_area", 3], ["object_detection", 8]]
# TASKS = [["semantic_segmentation", 21], ["lane_marking", 5], ["drivable_area", 3]]
tasks_name = "sem_lane_dri_det_WL_FL"
BACKBONE = "resnet34"  # one of resnet18, resnet34, resnet50, resnet101, resnet150.
IMAGE_WIDTH = 704
IMAGE_HEIGHT = 704
WEIGHT_DECAY = 0.0005
EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
WRITER = False  # Controlling the tensorboard
CONF_THRESHOLD = 0.5  # confidence of yolo model for the prediction
IOU_THRESH = 0.5
NMS_THRESH = 0.5
LOAD_MODEL_FILE = "models/" + tasks_name + "/" + tasks_name + "_172_.pth"
# LOAD_MODEL_FILE = "models/" + tasks_name + "/" + tasks_name + "5" + "_.pth"
SAVE_MODEL_FILE = "models/" + tasks_name + "/"  # folder to save models
SAVE_PATH = "predictions/" + tasks_name + "/"  # folder to save predictions
if not os.path.exists(SAVE_MODEL_FILE):
    os.makedirs(SAVE_MODEL_FILE)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
DATASET_DIR = "D:\Thesis\A2D2\Object_detection_bboxes/"  # labels main directory
IMG_DIR = "D:/Thesis/A2D2/Object_detection_bboxes/images/"
LABEL_DIR = ["1.sem_seg_masks/", "2.lane_masks/", "3.drivable_masks/",
             "4.det_labels/"]  # specific task labels folder insider labels main directory
# LABEL_DIR = ["1.sem_seg_masks/", "2.lane_masks/", "3.drivable_masks/"]

S = [IMAGE_WIDTH // 32, IMAGE_WIDTH // 16, IMAGE_WIDTH // 8]
ANCHORS = [[[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
           [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
           [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]]
scaled_anchors = (torch.tensor(ANCHORS) * (torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))).to(DEVICE)
train_transform, test_transform = transforms(IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT)


# training loop

def train_fn(train_loader, model, optimizer, loss_fn, GradScaler):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    mean_feature_loss = []
    idv_task_loss = torch.zeros(len(TASKS))

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y
        with torch.cuda.amp.autocast():
            out, features = model(x)
            loss, task_losses, feature_loss = loss_fn(out, y, features)
        optimizer.zero_grad()
        GradScaler.scale(loss).backward()
        GradScaler.step(optimizer)
        GradScaler.update()
        mean_loss.append(loss.item())
        mean_feature_loss.append(feature_loss)
        idv_task_loss += task_losses
        # update progress bar
        loop.set_postfix(loss=loss.item())
    mean_loss_value = sum(mean_loss) / len(mean_loss)
    mean_feature_loss_value = sum(mean_feature_loss) / len(mean_feature_loss)
    idv_task_loss = idv_task_loss / len(train_loader)
    print(f"Mean loss was {mean_loss_value}")
    print(f"Mean feature loss was {mean_feature_loss_value}")
    return mean_loss_value, idv_task_loss


def main():
    model = Multi_task_model(backbone=BACKBONE, in_channels=3, tasks=TASKS)
    model.to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if LOAD_MODEL:
        load_model(model, LOAD_MODEL_FILE, DEVICE=DEVICE)

    loss_fn = Multi_task_loss_fn(tasks=TASKS, scaled_anchors=scaled_anchors, DEVICE=DEVICE, WEIGHTED=True)

    GradScaler = torch.cuda.amp.GradScaler()

    prev_accuracy = 0

    train_loader, test_loader = get_loaders(
        train_csv="train.csv", test_csv="test.csv", dataset_dir=DATASET_DIR, img_dir=IMG_DIR,
        label_dir=LABEL_DIR, train_transform=train_transform, test_transform=test_transform,
        tasks=TASKS, anchors=ANCHORS, S=S, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    if WRITER:
        writer = SummaryWriter(f'tensorboard/{tasks_name}')

    for epoch in range(EPOCHS):
        model.train()
        print(f"EPOCH : {epoch}")
        mean_loss, task_losses = train_fn(train_loader, model, optimizer, loss_fn, GradScaler)
        if WRITER:
            writer.add_scalars("Losses", {task[0]: float(task_losses[i]) for i, task in enumerate(TASKS)},
                               global_step=epoch)
        save_model(model, SAVE_MODEL_FILE + str(epoch) + "_.pth")                         # save model after every epoch
        task_accuracies = check_accuracies(model, loader=test_loader, tasks=TASKS, anchors=ANCHORS,
                                           nms_threshold=NMS_THRESH, DEVICE=DEVICE,
                                           iou_threshold=IOU_THRESH, conf_threshold=CONF_THRESHOLD,
                                           check_map=True,  # if epoch > 19 and epoch % 5 == 0 else False,
                                           class_wise_dice_score=True)
        test_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
        if WRITER:
            writer.add_scalars("Accuracies", {key: task_accuracies[key] for key in task_accuracies.keys()},
                               global_step=epoch)
        text_file = open(SAVE_PATH + tasks_name + ".txt", "w")   # text file to write test image ID and models accuracy on test image
        save_some_examples(model, test_loader, save_path=SAVE_PATH, epoch=epoch, tasks=TASKS,
                           anchors=scaled_anchors, iou_threshold=IOU_THRESH, threshold=0.65, nms_threshold=0.2,
                           DEVICE=DEVICE, text_file=text_file)  # save some predictions
        text_file.close()
        if test_accuracy > prev_accuracy:
            save_model(model, str(test_accuracy) + tasks_name + "model")  # saving the best model
            prev_accuracy = test_accuracy


if __name__ == "__main__":
    main()
