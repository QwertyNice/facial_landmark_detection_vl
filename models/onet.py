import torch

from torch import nn

from utils.utils import weights_init


class ONet(nn.Module):
    def __init__(self, input_size, total_landmarks=68):
        super(ONet, self).__init__()

        self.input_size = input_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )

        self.conv4 = nn.Linear(self.configurate_forward(), 256)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Linear(256, 2 * total_landmarks)

        self.apply(weights_init)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        landmarks = self.conv5(x)
        return landmarks

    def configurate_forward(self):
        x = torch.zeros([1, 3, self.input_size, self.input_size])
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x.size()[1]
