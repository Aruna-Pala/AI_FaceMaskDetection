import torch.nn as nn
import torch.nn.functional as F
import torch


class Face_Mask_CNN3(nn.Module):

    def __init__(self, num_classes=3):
        super(Face_Mask_CNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=46, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=46, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=64 * 64 * 64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = x.view(-1, 64 * 64 * 64)

        x = self.fc1(x)

        return torch.log_softmax(x, dim=1)
