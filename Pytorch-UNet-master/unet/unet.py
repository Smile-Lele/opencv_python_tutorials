import torch
import torch.nn as nn
import torchvision.models as models


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

        self.upSampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        x = torch.cat([inputs1, self.upSampling(inputs2)], dim=1)
        x = self.conv_relu(x)

        return x


class Unet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(Unet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = 3
        # 主干特征提取网络，based on VGG16
        # the channels of input must be 3
        # coz, (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg = models.vgg16(pretrained)
        del self.vgg.avgpool
        del self.vgg.classifier

        # 加强特征提取网络，上采样
        self.up4 = UnetUp(1024, 512)  # 64,64,512
        self.up3 = UnetUp(768, 256)  # 128,128,256
        self.up2 = UnetUp(384, 128)  # 256,256,128
        self.up1 = UnetUp(192, 64)  # 512,512,64
        self.conv = nn.Conv2d(64, self.n_classes, kernel_size=(1, 1))

    def forward(self, x):
        # down sampling
        feat1 = self.vgg.features[:4](x)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        # up sampling
        up4 = self.up4(feat4, feat5)
        up3 = self.up3(feat3, up4)
        up2 = self.up2(feat2, up3)
        up1 = self.up1(feat1, up2)
        out = self.conv(up1)

        return out

    def freeze_backbone(self):
        for param in self.vgg.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    unet = Unet(2)
    print(unet)

    exit(0)

    import cv2 as cv

    image = cv.imread('data_/dataset/imgs/1_0.png', cv.IMREAD_COLOR)
    image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

    image_tensor = torch.from_numpy(image).to(torch.float32)
    image_tensor.transpose_(2, 0)
    image_tensor.unsqueeze_(0)
    # print(image_tensor.size())
    pred = unet(image_tensor)
    pred.squeeze_(0)
    pred.transpose_(0, 2)
    y = pred.detach().numpy()
    print(y.shape)

    from matplotlib import pyplot as plt

    plt.imshow(y[:, :, 1])
    plt.show()
