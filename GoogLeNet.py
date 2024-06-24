# Imports
import torch
import torch.nn as nn

# Implementatiom without auxiliary classifiers
class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7,
                               stride=2, padding=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3,
                               stride=1, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception_block(in_channels=192, out_1x1=64, reduce_3x3=96, out_3x3=128,
                                           reduce_5x5=16, out_5x5=32, out_1x1pool=32)

        self.inception3b = Inception_block(in_channels=256, out_1x1=128, reduce_3x3=128, out_3x3=192,
                                           reduce_5x5=32, out_5x5=96, out_1x1pool=64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(in_channels=480, out_1x1=192, reduce_3x3=96, out_3x3=208,
                                           reduce_5x5=16, out_5x5=48, out_1x1pool=64)

        self.inception4b = Inception_block(in_channels=512, out_1x1=160, reduce_3x3=112, out_3x3=224,
                                           reduce_5x5=24, out_5x5=64, out_1x1pool=64)

        self.inception4c = Inception_block(in_channels=512, out_1x1=128, reduce_3x3=128, out_3x3=256,
                                           reduce_5x5=24, out_5x5=64, out_1x1pool=64)

        self.inception4d = Inception_block(in_channels=512, out_1x1=112, reduce_3x3=144, out_3x3=288,
                                           reduce_5x5=32, out_5x5=64, out_1x1pool=64)

        self.inception4e = Inception_block(in_channels=528, out_1x1=256, reduce_3x3=160, out_3x3=320,
                                           reduce_5x5=32, out_5x5=128, out_1x1pool=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(in_channels=832, out_1x1=256, reduce_3x3=160, out_3x3=320,
                                           reduce_5x5=32, out_5x5=128, out_1x1pool=128)

        self.inception5b = Inception_block(in_channels=832, out_1x1=384, reduce_3x3=192, out_3x3=384,
                                           reduce_5x5=48, out_5x5=128, out_1x1pool=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(p=0.4)

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_1x1pool):
        super().__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, reduce_3x3, kernel_size=1),
            conv_block(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, reduce_5x5, kernel_size=1),
            conv_block(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


if __name__ == "__main__":
    BATCH_SIZE = 5
    x = torch.randn(BATCH_SIZE, 3, 224, 224)
    model = GoogLeNet(num_classes=1000)
    output = model(x)
    print(output.shape)
    assert output.shape == torch.Size([BATCH_SIZE, 1000])
