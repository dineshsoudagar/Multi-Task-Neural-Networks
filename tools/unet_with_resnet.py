import torch
import torchvision.models.resnet as models
import torch.nn as nn
import torchvision.transforms.functional as TF


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


class UNET_Resnet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=20, features=None, backbone=None,
    ):
        super(UNET_Resnet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        if backbone is None :
            self.downs = nn.ModuleList()
        else:
            self.downs = backbone
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        if backbone is None:
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature

        # Up part of UNET
        for i in range(1,len(features)):
            in_feature = features[-i]
            out_feature = features[-i-1]
            self.ups.append(
                nn.ConvTranspose2d(
                    in_feature, out_feature, kernel_size=(2, 2), stride=(2, 2),
                )
            )
            self.ups.append(DoubleConv(out_feature*2, out_feature))

        self.bottleneck = DoubleConv(features[-2], features[-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=(1, 1))

    def forward(self, x):
        skip_connections = []
        x = self.downs.conv1(x)
        x = self.downs.bn1(x)
        x = self.downs.relu(x)
        skip_connections.append(x)  # skip1

        x = self.downs.maxpool(x)
        x = self.downs.layer1(x)
        skip_connections.append(x)  # skip2

        x = self.downs.layer2(x)
        skip_connections.append(x)  # skip3

        x = self.downs.layer3(x)
        skip_connections.append(x)  # skip4

        x = self.downs.layer4(x)
        skip_connections.append(x)  # skip5

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def prepare_resnet_model(backbone = "resnet34"):

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


def UNET_with_resnet(backbone=None, in_channels=3, out_channels=20):

    model, features = prepare_resnet_model(backbone=backbone)
    model = UNET_Resnet(backbone=model, in_channels=in_channels, out_channels=out_channels, features=features)

    return model


def test():

    model = UNET_with_resnet(backbone="resnet34", in_channels=3, out_channels=20)
    x = torch.randn(2, 3, 1024, 720)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()
