"""
Coverting A2D2 Json 2D BBoxes to Yolo format
"""

import glob
import json
import cv2

A2D2_bbox_categories = {
    "Car": 0,
    "Pedestrian": 1,
    "Truck": 2,
    "VanSUV": 3,
    "Cyclist": 4,
    "Bus": 5,
    "MotorBiker": 6,
    "Bicycle": 7,
    "UtilityVehicle": 8,
    "Motorcycle": 9,
    "CaravanTransporter": 10,
    "Animal": 11,
    "Trailer": 12,
    "EmergencyVehicle": 13,
}


def check_bbox_values(box):
    # function to make sure the box coordinates are within the image size

    img_w_h = [0, 1919, 1207, 1919, 1207]

    for i in range(1, 5):
        if box[i] < 0:
            box[i] = 1
        elif box[i] > img_w_h[i]:
            box[i] = img_w_h[i]

    return box


def display_bboxes_on_image(image, bboxes, image_file_name):
    for box in bboxes:
        cv2.rectangle(image, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (36, 255, 12), 5)
    cv2.imshow("image", image)
    cv2.imwrite(image_file_name, image)
    cv2.waitKey(0)


def convert_to_xywh_format(bbox):
    x = ((bbox[1] + bbox[3]) / 2) * (1 / 1920)
    y = ((bbox[2] + bbox[4]) / 2) * (1 / 1208)
    w = (bbox[3] - bbox[1]) / 1920
    h = (bbox[4] - bbox[2]) / 1208

    return [bbox[0], x, y, w, h]


# path for all json files
json_files_list = glob.glob("D:\Thesis\A2D2\Object_detection_bboxes/bboxes/*.json")
image = "D:/Thesis/A2D2/Object_detection_bboxes/images/"

for file in json_files_list:
    f = open(file)
    data = json.load(f)
    file_name = file.rsplit("/")[-1].rsplit("\\")[-1].rsplit(".")[0]
    text_file = open(file_name + ".txt", 'w')
    image_file_name_split = file_name.rsplit("_")

    # Load Image to display bboxes

    # image_file_name = image_file_name_split[0] + "_" + "camera" + "_" + image_file_name_split[2] + "_" + image_file_name_split[3] + ".png"
    # image = cv2.imread("D:/Thesis/A2D2/Object_detection_bboxes/images/" + (image_file_name))

    # bboxes = []

    for i in range(len(data)):
        box = "box_" + str(i)
        bbox = data[box]["2d_bbox"]
        bbox_original = bbox
        category_id = A2D2_bbox_categories[data[box]["class"]]  # integer

        # box coordinates
        x1 = int(bbox[0])  # bottom left
        y1 = int(bbox[1])  # bottom left
        x2 = int(bbox[2])  # top right
        y2 = int(bbox[3])  # top right

        bbox = [category_id, x1, y1, x2, y2]
        bbox = check_bbox_values(bbox)
        # bboxes.append(bbox)

        bbox = convert_to_xywh_format(bbox)

        for i in range(1, 5):
            if bbox[i] > 1 or bbox[i] <= 0:
                print(bbox_original)
                print(bbox)

                print(file_name)

        label = str((bbox[0])) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(bbox[4])
        text_file.write(label)
        text_file.write("\n")

    # display_bboxes_on_image(image,bboxes,image_file_name)
