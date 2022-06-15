import zipfile

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid


class DataLoaderHelper:
    random_seed = 48
    torch.manual_seed(random_seed)

    def extractZip(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(path="./data")
        return "./data/Data Set"

    def createTransforms(self, data_set_path):
        transF = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        data_set = ImageFolder(data_set_path, transform=transF)
        return data_set

    def createDataLoader(self, data_set, batch_size):
        return DataLoader(data_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def splitData(self, training_data_set, batch_size):
        test_size = int(len(training_data_set) * 0.25)
        train_size = len(training_data_set) - test_size
        train_data_set, test_data_set = random_split(training_data_set, [train_size, test_size])
        print(len(train_data_set), len(test_data_set))
        train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_data_loader = DataLoader(test_data_set, batch_size * 2, num_workers=4, pin_memory=True)
        return train_data_loader, test_data_loader

    def showBatch(self, dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break
        plt.show()

    def printDataSetInfo(self, data_set, data_loader):
        image, label = data_set[0]
        print(image.shape, label)
        categories = data_set.classes
        print(categories)
        self.showBatch(data_loader)
