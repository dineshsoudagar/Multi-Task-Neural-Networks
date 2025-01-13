"""
Implementation of YOLOv3 architecture with resnet
"""

import torch
import torch.nn as nn
import torchvision.models.resnet as models

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (512, 2, 2),  # changed from (512,1,1) downsampling to match the first output size (32,16)
    (1024, 3, 1),
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


# To remove resnet last layers
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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

    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    resnet.avgpool = Identity()
    resnet.fc = Identity()

    return resnet, features


class Y0l0v3_resnet(nn.Module):
    def __init__(self, num_classes=20, backbone="resnet50"):
        super().__init__()
        self.resnet, self.features = prepare_resnet_model(backbone=backbone)
        self.num_classes = num_classes
        self.in_channels = self.features[-2]
        self.factor = self.features[-2] // 512
        self.prediction_layers = self.create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        skip_connections = []

        # forward pass through resnet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        skip_connections.append(x)  # skip1

        x = self.resnet.layer4(x)
        skip_connections.append(x)  # skip2

        # forward pass through scale prediction

        for layer in self.prediction_layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, skip_connections[-1]], dim=1)
                skip_connections.pop()

        return outputs

    def create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.features[-2]

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
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2), )
                    in_channels = in_channels + in_channels * 2 * self.factor

        return layers


if __name__ == "__main__":
    num_classes = 14
    IMAGE_SIZE = 704
    model = Y0l0v3_resnet(num_classes=num_classes, backbone="resnet34")
    print(model)
    input_img = torch.randn((2, 3, 704, 704))
    out = model(input_img)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    assert model(input_img)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(input_img)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(input_img)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
