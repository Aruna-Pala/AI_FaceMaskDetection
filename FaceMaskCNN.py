import torch
from torch import nn
import torch.nn.functional as F


class FaceMaskCNN4(nn.Module):

    def __init__(self, num_classes=5):
        super(FaceMaskCNN4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=20)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=28)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=28, out_channels=36, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=36)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=36, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=40)

        self.drop1 = nn.Dropout(0.25)

        self.conv6 = nn.Conv2d(in_channels=40, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=48)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.drop2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=8 * 8 * 48, out_features=64)

        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.pool1(self.bn2(self.conv2(x))))

        x = F.relu(self.pool2(self.bn3(self.conv3(x))))

        x = F.relu(self.pool3(self.bn4(self.conv4(x))))

        x = self.drop1(x)

        x = F.relu(self.bn5(self.conv5(x)))

        x = F.relu(self.pool4(self.bn6(self.conv6(x))))

        x = self.drop2(x)

        x = x.view(-1, 8 * 8 * 48)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)

    def getDevice(self):
        if torch.cuda.is_available():
            return 'cuda'
        try:
            if torch.backends.mps.is_available():
                return 'mps'
        except:
            return 'cpu'

# model3 = FaceMaskCNN4().to(device)

# print(model3)
