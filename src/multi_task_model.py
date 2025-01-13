import torch
import torchvision.models.resnet as models
import torch.nn as nn
import torchvision.transforms.functional as TF
from utils import load_model

"""
config file
tuple : (out_channels, kernel, stride)
"S" : Scale prediction layer
"U" : Up sample layer
"D" : Downsampling , not in original yolo v3 
"""

config = [
    (512, 2, 2),
    (1024, 3, 1),
    # "D",  # added to match the first output size (img_width//32,img_height//32)
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def create_heads(task=None, head=None, features=None, num_classes=20):
    head.append(DoubleConv(features[-2], features[-1]))
    if task != "object_detection":
        for i in range(1, len(features)):
            in_feature = features[-i]
            out_feature = features[-i - 1]
            head.append(
                nn.ConvTranspose2d(
                    in_feature, out_feature, kernel_size=(2, 2), stride=(2, 2),
                )
            )
            head.append(DoubleConv(out_feature * 2, out_feature))
    head.append(nn.Conv2d(features[0], num_classes, kernel_size=(1, 1)))

    return head


def create_yolo_conv_layers(in_channels=None, num_classes=20, head=None):
    layers = head
    in_channels = in_channels
    factor = in_channels // 512

    for module in config:
        if isinstance(module, tuple):
            out_channels, kernel_size, stride = module
            layers.append(
                CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0,
                )
            )
            in_channels = out_channels

        elif isinstance(module, list):
            num_repeats = module[1]
            layers.append(ResidualBlock(in_channels, num_repeats=num_repeats, ))

        elif isinstance(module, str):
            if module == "S":
                layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ScalePrediction(in_channels // 2, num_classes=num_classes),
                ]
                in_channels = in_channels // 2

            elif module == "U":
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels + in_channels * 2 * factor  # because resnet layer 4 has 2048 channels

            elif module == "D":
                layers.append(nn.MaxPool2d(2))

    return layers


def prepare_resnet_model(backbone="resnet34"):
    resnet = None
    features = None

    if backbone == "resnet18":
        resnet = models.resnet18(pretrained=True)
        features = [64, 64, 128, 256, 512, 1024]

    elif backbone == "resnet34":
        resnet = models.resnet34(pretrained=True)
        features = [64, 64, 128, 256, 512, 1024]

    elif backbone == "resnet50":
        resnet = models.resnet50(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone == "resnet101":
        resnet = models.resnet101(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone == "resnet152":
        resnet = models.resnet152(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone is None:
        features = [64, 128, 256, 512, 1024]

    resnet.avgpool = Identity()
    resnet.fc = Identity()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

    return resnet, features


class Multi_task_model(nn.Module):
    def __init__(self, in_channels=3, backbone=None, tasks=None):
        super(Multi_task_model, self).__init__()

        """
        in_channels : input channel dimension
        backbone : one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        tasks : list [["task name", no. of classes],["task name", no. of classes]....]
        task names : "semantic_segmentation", "lane_marking", "drivable_area", "object_detection"
        returns : model outputs, features for feature loss
        """

        if tasks is None:
            raise ValueError("tasks are not mentioned")
        self.in_channels = in_channels
        self.resnet, self.features = prepare_resnet_model(backbone=backbone)  # creating tail of multi-task model
        self.pool = nn.MaxPool2d(2, 2)
        self.tasks = tasks
        self.heads = []
        for task in self.tasks:
            if task[0] == "semantic_segmentation":  # semantic_segmentation head
                self.segmentation = nn.ModuleList()
                self.heads.append(create_heads(
                    task=task,
                    head=self.segmentation,
                    num_classes=task[1],
                    features=self.features,
                ))
            elif task[0] == "lane_marking":  # lane_marking head
                self.lane = nn.ModuleList()
                self.heads.append(create_heads(
                    task=task,
                    head=self.lane,
                    num_classes=task[1],
                    features=self.features,
                ))
            elif task[0] == "drivable_area":  # drivable area head
                self.drivable = nn.ModuleList()
                self.heads.append(create_heads(
                    task=task,
                    head=self.drivable,
                    num_classes=task[1],
                    features=self.features,
                ))
            elif task[0] == "object_detection":  # object_detection head
                self.object_detection = nn.ModuleList()
                self.heads.append(create_yolo_conv_layers(
                    in_channels=self.features[-2],
                    num_classes=task[1],
                    head=self.object_detection,
                ))

    def forward(self, x):
        outputs = {}
        features_track = {task[0]: False for task in self.tasks}
        features = {}
        skip_connections = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        skip_connections.append(x)  # skip1

        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        skip_connections.append(x)  # skip2

        x = self.resnet.layer2(x)
        skip_connections.append(x)  # skip3

        x = self.resnet.layer3(x)
        skip_connections.append(x)  # skip4, also yolo skip2

        x = self.resnet.layer4(x)
        skip_connections.append(x)  # skip5, also yolo skip1

        resnet_output = x

        bottle_neck_input = self.pool(resnet_output)

        skip_connections = skip_connections[::-1]

        for i, task in enumerate(self.tasks):
            if task[0] != "object_detection":
                for idx in range(len(self.heads[i])):
                    if idx == 0:
                        x = self.heads[i][idx](bottle_neck_input)
                        if not features_track[task[0]]:
                            features[task[0]] = x
                            features_track[task[0]] = True

                    elif idx % 2 == 0:
                        skip_connection = skip_connections[(idx // 2) - 1]
                        if x.shape != skip_connection.shape:
                            x = TF.resize(x, size=skip_connection.shape[2:])
                        concat_skip = torch.cat((skip_connection, x), dim=1)
                        x = self.heads[i][idx](concat_skip)

                    else:
                        x = self.heads[i][idx](x)

                outputs[task[0]] = x

            elif task[0] == "object_detection":
                idx = 1
                skip_connections_yolo = skip_connections[::-1][3:5]
                yolo_outputs = []
                x = resnet_output
                for layer in self.heads[i]:
                    if isinstance(layer, ScalePrediction) and idx == 1:
                        yolo_outputs.append(layer(x))
                        continue
                    elif isinstance(layer, ScalePrediction):
                        yolo_outputs.append(layer(x))
                        continue

                    x = layer(x)
                    if not features_track[task[0]] and x.shape[1] == 1024:
                        features[task[0]] = x
                        features_track[task[0]] = True

                    if isinstance(layer, nn.Upsample):
                        x = torch.cat([x, skip_connections_yolo[-idx]], dim=1)
                        idx += 1

                outputs[task[0]] = yolo_outputs

        return outputs, features


def load_backbone(tasks=None, resnet="resnet34", tasks_name=None, frozen=True):
    if tasks is None:
        backbone = prepare_resnet_model(backbone=resnet)
    else:
        backbone = Multi_task_model(tasks=tasks, backbone=resnet)
        load_model(backbone,
                   name="D:\Thesis/5.multi_task_model_A2D2_pillow\models/" + tasks_name + "/" + tasks_name + ".pth")

    if frozen:
        for param in backbone.resnet.parameters():
            param.requires_grad = False

    return backbone


def model_with_frozen_backbone(main_task, backbone_tasks, backbone_name=None, resnet="resnet34"):
    backbone = load_backbone(tasks=backbone_tasks, tasks_name=backbone_name, resnet=resnet)
    model = Multi_task_model(tasks=main_task, backbone=resnet)

    if backbone_tasks is None:
        model.resnet = backbone.resnet
    else:
        model.resnet = backbone

    return model


def test():
    tasks = [["semantic_segmentation", 20], ["object_detection", 16], ["lane_marking", 5], ["drivable_area", 3]]
    model = Multi_task_model(backbone="resnet34", in_channels=3, tasks=tasks)

    # backbone_tasks = [["semantic_segmentation", 20], ["lane_marking", 5], ["drivable_area", 3]]
    # main_task = [["object_detection", 16]]
    # model = model_with_frozen_backbone(main_task=main_task, backbone_tasks=backbone_tasks, resnet="resnet34",
    #                                   backbone_name="sem_lane_dri.pth")
    x = torch.randn(2, 3, 704, 704)
    print(model)
    y, features = model(x)
    print(len(features))
    # print(y["semantic_segmentation"].shape)
    # print(len(features))
    # print(features["semantic_segmentation"].shape)
    # print(features["drivable_area"].shape)

    for task in tasks:
        print(task[0])
        if task[0] != "object_detection":
            print(y[task[0]].shape)
            print(features[task[0]].shape)
        elif task[0] == "object_detection":
            print(features[task[0]].shape)
            for j in range(3):
                print(y[task[0]][j].shape)
            # for i in range(len(y)):
            #   if len(y[i]) == 3:
            #       for j in range(3):
            #           print(y[i][j].shape)
            #   else:
            #       print(y[i][0].shape)


if __name__ == '__main__':
    test()
