from turtle import forward
import torch
import torch.nn as nn

 
architecture_config = [
    # Tuple = (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # list = [tuples, num_repeats]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                        out_channels=out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.architecture = architecture_config
        self.darknet = self._create_darknet(self.architecture)
        self.fc = self._create_fc(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def _create_darknet(self, architecture):
        layers = []
        inChannels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                outChannels = x[1]
                kernel_size = x[0]
                stride = x[2]
                padding = x[3]
                layers += [CNNBlock(in_channels=inChannels,
                                out_channels=outChannels, kernel_size=kernel_size,
                                stride = stride, padding = padding)]
                inChannels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels=inChannels, 
                            out_channels=conv1[1],
                            kernel_size = conv1[0], 
                            stride=conv1[2], 
                            padding=conv1[3])
                        ]
                    inChannels = conv1[1]
                    layers += [
                        CNNBlock(
                            in_channels=inChannels, 
                            out_channels=conv2[1],
                            kernel_size = conv2[0], 
                            stride=conv2[2], 
                            padding=conv2[3])
                        ]
                    inChannels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fc(self, Split_size, num_box, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * Split_size * Split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, Split_size*Split_size*(num_classes + num_box * 5)),
        )   

if __name__ == "main":
    def test(splitSize = 7, numBoxes = 2, numClasses = 20):
        model = YOLOv1(in_channels=3, Split_size = splitSize, num_box = numBoxes, num_classes = numClasses)
        x = torch.rand((2, 3, 448, 448))
        print(model(x).shape)
    test() 




