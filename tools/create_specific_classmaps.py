import glob
from PIL import ImageColor
import cv2
import numpy as np
from tqdm import tqdm

categories_color_ids = {
    "Car 1": "#ff0000",
    "Car 2": "#c80000",
    "Car 3": "#960000",
    "Car 4": "#800000",
    "Bicycle 1": "#b65906",
    "Bicycle 2": "#963204",
    "Bicycle 3": "#5a1e01",
    "Bicycle 4": "#5a1e1e",
    "Pedestrian 1": "#cc99ff",
    "Pedestrian 2": "#bd499b",
    "Pedestrian 3": "#ef59bf",
    "Truck 1": "#ff8000",
    "Truck 2": "#c88000",
    "Truck 3": "#968000",
    "Small vehicles 1": "#00ff00",
    "Small vehicles 2": "#00c800",
    "Small vehicles 3": "#009600",
    "Traffic signal 1": "#0080ff",
    "Traffic signal 2": "#1e1c9e",
    "Traffic signal 3": "#3c1c64",
    "Traffic sign 1": "#00ffff",
    "Traffic sign 2": "#1edcdc",
    "Traffic sign 3": "#3c9dc7",
    "Utility vehicle 1": "#ffff00",
    "Utility vehicle 2": "#ffffc8",
    "Sidebars": "#e96400",
    "Speed bumper": "#6e6e00",
    "Curbstone": "#808000",
    "Solid line": "#ffc125",
    "Irrelevant signs": "#400040",
    "Road blocks": "#b97a57",
    "Tractor": "#000064",
    "Non-drivable street": "#8b636c",
    "Zebra crossing": "#d23273",
    "Obstacles / trash": "#ff0080",
    "Poles": "#fff68f",
    "RD restricted area": "#960096",
    "Animals": "#ccff99",
    "Grid structure": "#eea2ad",
    "Signal corpus": "#212cb1",
    "Drivable cobblestone": "#b432b4",
    "Electronic traffic": "#ff46b9",
    "Slow drive area": "#eee9bf",
    "Nature object": "#93fdc2",
    "Parking area": "#9696c8",
    "Sidewalk": "#b496c8",
    "Ego car": "#48d1cc",
    "Painted driv. instr.": "#c87dd2",
    "Traffic guide obj.": "#9f79ee",
    "Dashed line": "#8000ff",
    "RD normal street": "#ff00ff",
    "Sky": "#87ceff",
    "Buildings": "#f1e6ff",
    "Blurred area": "#60458f",
    "Rain dirt": "#352e52",

}

categories_class_ids = {
    "Car 1": 0,
    "Car 2": 0,
    "Car 3": 0,
    "Car 4": 0,
    "Bicycle 1": 1,
    "Bicycle 2": 1,
    "Bicycle 3": 1,
    "Bicycle 4": 1,
    "Pedestrian 1": 2,
    "Pedestrian 2": 2,
    "Pedestrian 3": 2,
    "Truck 1": 3,
    "Truck 2": 3,
    "Truck 3": 3,
    "Small vehicles 1": 4,
    "Small vehicles 2": 4,
    "Small vehicles 3": 4,
    "Traffic signal 1": 5,
    "Traffic signal 2": 5,
    "Traffic signal 3": 5,
    "Traffic sign 1": 6,
    "Traffic sign 2": 6,
    "Traffic sign 3": 6,
    "Utility vehicle 1": 7,
    "Utility vehicle 2": 7,
    "Sidebars": 8,
    "Speed bumper": 9,
    "Curbstone": 10,
    "Solid line": 11,
    "Irrelevant signs": 12,
    "Road blocks": 13,
    "Tractor": 14,
    "Non-drivable street": 15,
    "Zebra crossing": 16,
    "Obstacles / trash": 17,
    "Poles": 18,
    "RD restricted area": 19,
    "Animals": 20,
    "Grid structure": 21,
    "Signal corpus": 22,
    "Drivable cobblestone": 23,
    "Electronic traffic": 24,
    "Slow drive area": 25,
    "Nature object": 26,
    "Parking area": 27,
    "Sidewalk": 28,
    "Ego car": 29,
    "Painted driv. instr.": 30,
    "Traffic guide obj.": 31,
    "Dashed line": 32,
    "RD normal street": 33,
    "Sky": 34,
    "Buildings": 35,
    "Blurred area": 36,
    "Rain dirt": 37,

}

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

color_to_class_id = {
    "#ff0000": 0,
    "#c80000": 0,
    "#960000": 0,
    "#800000": 0,
    "#b65906": 1,
    "#963204": 1,
    "#5a1e01": 1,
    "#5a1e1e": 1,
    "#cc99ff": 2,
    "#bd499b": 2,
    "#ef59bf": 2,
    "#ff8000": 3,
    "#c88000": 3,
    "#968000": 3,
    "#00ff00": 4,
    "#00c800": 4,
    "#009600": 4,
    "#0080ff": 5,
    "#1e1c9e": 5,
    "#3c1c64": 5,
    "#00ffff": 6,
    "#1edcdc": 6,
    "#3c9dc7": 6,
    "#ffff00": 7,
    "#ffffc8": 7,
    "#e96400": 8,
    "#6e6e00": 9,
    "#808000": 10,
    "#ffc125": 11,
    "#400040": 12,
    "#b97a57": 13,
    "#000064": 14,
    "#8b636c": 15,
    "#d23273": 16,
    "#ff0080": 17,
    "#fff68f": 18,
    "#960096": 19,
    "#ccff99": 20,
    "#eea2ad": 21,
    "#212cb1": 22,
    "#b432b4": 23,
    "#ff46b9": 24,
    "#eee9bf": 25,
    "#93fdc2": 26,
    "#9696c8": 27,
    "#b496c8": 28,
    "#48d1cc": 29,
    "#c87dd2": 30,
    "#9f79ee": 31,
    "#8000ff": 32,
    "#ff00ff": 33,
    "#87ceff": 34,
    "#f1e6ff": 35,
    "#60458f": 36,
    "#352e52": 37,
}

class_id_to_category = {
    0: "Car 1",
    1: "Bicycle 1",
    2: "Pedestrian 1",
    3: "Truck 1",
    4: "Small vehicles 1",
    5: "Traffic signal 1",
    6: "Traffic sign 1",
    7: "Utility vehicle 1",
    8: "Sidebars",
    9: "Speed bumper",
    10: "Curbstone",
    11: "Solid line",
    12: "Irrelevant signs",
    13: "Road blocks",
    14: "Tractor",
    15: "Non-drivable street",
    16: "Zebra crossing",
    17: "Obstacles / trash",
    18: "Poles",
    19: "RD restricted area",
    20: "Animals",
    21: "Grid structure",
    22: "Signal corpus",
    23: "Drivable cobblestone",
    24: "Electronic traffic",
    25: "Slow drive area",
    26: "Nature object",
    27: "Parking area",
    28: "Sidewalk",
    29: "Ego car",
    30: "Painted driv. instr.",
    31: "Traffic guide obj.",
    32: "Dashed line",
    33: "RD normal street",
    34: "Sky",
    35: "Buildings",
    36: "Blurred area",
    37: "Rain dirt",
}

class_id_to_color = {
    0: "#ff0000",
    # 0: "#800000",
    1: "#b65906",
    # 1: "#963204",
    # 1: "#5a1e01",
    # 1: "#5a1e1e",
    2: "#cc99ff",
    # 2: "#bd499b",
    # 2: "#ef59bf",
    3: "#ff8000",
    # 3: "#c88000",
    # 3: "#968000",
    4: "#00ff00",
    # 4: "#00c800",
    # 4: "#009600",
    5: "#0080ff",
    # 5: "#1e1c9e",
    # 5: "#3c1c64",
    6: "#00ffff",
    # 6: "#1edcdc",
    # 6: "#3c9dc7",
    7: "#ffff00",
    # 7: "#ffffc8",
    8: "#e96400",
    9: "#6e6e00",
    10: "#808000",
    11: "#ffc125",
    12: "#400040",
    13: "#b97a57",
    14: "#000064",
    15: "#8b636c",
    16: "#d23273",
    17: "#ff0080",
    18: "#fff68f",
    19: "#960096",
    20: "#ccff99",
    21: "#eea2ad",
    22: "#212cb1",
    23: "#b432b4",
    24: "#ff46b9",
    25: "#eee9bf",
    26: "#93fdc2",
    27: "#9696c8",
    28: "#b496c8",
    29: "#48d1cc",
    30: "#c87dd2",
    31: "#9f79ee",
    32: "#8000ff",
    33: "#ff00ff",
    34: "#87ceff",
    35: "#f1e6ff",
    36: "#60458f",
    37: "#352e52",
}

rgb_avg_to_class_id = {65025: 0, 40000: 0, 22500: 0, 16384: 0, 33581: 1, 22758: 1, 8252: 1, 8310: 1, 42891: 2, 36396: 2,
                       57948: 2, 65665: 3, 40640: 3, 23140: 3, 1275: 4, 1000: 4, 750: 4, 1150: 5, 1356: 5, 3940: 5,
                       1785: 6, 2440: 6, 4783: 6, 66300: 7, 66700: 7, 54789: 8, 12650: 9, 17024: 10, 66064: 11,
                       4224: 12, 35009: 13, 200: 14, 20032: 15, 44580: 16, 65281: 17, 66541: 18, 22800: 19, 43197: 20,
                       57800: 21, 1663: 22, 33010: 23, 65745: 24, 58191: 25, 23262: 26, 23650: 27, 33550: 28, 6637: 29,
                       41045: 30, 26362: 31, 16894: 32, 65535: 33, 19765: 34, 59741: 35, 9847: 36, 3203: 37}


# print(len(rgb_avg_to_class_id))

# for key in color_to_class_id.keys():
#    color_id = key
#    rgb_value = ImageColor.getcolor(color_id,"RGB")
#    r, g, b = rgb_value
#    rgb_value_new = (r*r+g*5+b*2)
#    rgb_avg_to_class_id[rgb_value_new] = color_to_class_id[color_id]
#


def create_class_mask(image):
    img_original = np.array(image, dtype=np.int32)
    img_original_rxr = (np.square(img_original[:, :, 2]))
    img_original_gx5 = img_original[:, :, 1] * 5
    img_original_bx2 = img_original[:, :, 0] * 2

    img_original_sum = img_original_rxr + img_original_gx5 + img_original_bx2
    class_mask = np.zeros_like(img_original_sum)

    avg_color_values = np.unique(img_original_sum)

    for color in avg_color_values:
        class_mask = np.where(img_original_sum == color, rgb_avg_to_class_id[color], class_mask)
    return class_mask


def mask_to_colormap(image):
    mask = np.expand_dims(image, axis=2)
    mask = np.repeat((mask), 3, 2)

    color_map = np.ones_like(mask) * 255
    class_ids = np.unique(mask)

    for class_id in class_ids:

        if class_id != 255:
            rgb_value = ImageColor.getcolor(class_id_to_color[class_id], "RGB")
            r, g, b = rgb_value
            color_map = np.where(mask == [[class_id, class_id, class_id]], [[b, g, r]], color_map)
    return color_map


def create_specific_class_mask(mask, specific_class=None):
    mask_original = mask
    mask = np.ones_like(mask) * 255  # that means the background class will have value of 255

    for class_id in np.unique(mask_original):
        if class_id in specific_class.keys():
            mask = np.where(mask_original == class_id, specific_class[class_id], mask)
    return mask


# list of original class maps with all the classes
files_list = glob.glob("D:\Thesis\A2D2\Object_detection_bboxes/sem_seg_masks_all_classes/*.png")

save_path = "D:\Thesis\A2D2\Object_detection_bboxes/2.lane_masks_2/"


# specify which class from original class map to be changed , original class : new class
# lane_specific_classes = {9: 0, 11: 1, 16: 2, 30: 3, 32: 4} # 9 is speedbumber , not part of lane marking
lane_specific_classes = {11: 0, 16: 1, 30: 2, 32: 3}
Drivable_specific_classes = {9: 1, 10: 0, 11: 1, 15: 0, 16: 1, 19: 0, 23: 1, 25: 1, 30: 1, 32: 1, 33: 1, 37: 1, 28: 0}
sem_seg_specific_classes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 3, 8: 7, 9: 8, 10: 8, 11: 8,
                            13: 9, 14: 3, 16: 8, 18: 10, 20: 11, 21: 12, 22: 13, 23: 8, 25: 8, 26: 14,
                            27: 15, 28: 16, 30: 8, 31: 17, 32: 8, 33: 8, 34: 18, 35: 19,
                            }

loop = tqdm(files_list, leave=True)

for file in loop:
    class_mask_file_name = file.rsplit("\\")[-1]
    mask_original = cv2.imread(file)
    specific_class_mask = create_specific_class_mask(mask_original, lane_specific_classes)
    cv2.imwrite(save_path + class_mask_file_name, specific_class_mask)
