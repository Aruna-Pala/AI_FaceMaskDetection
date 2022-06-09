import os
import torch
import zipfile

from matplotlib import pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

if __name__ == '__main__':
    with zipfile.ZipFile("./Data_Set.zip", 'r') as zip:
        zip.extractall(path="./data")

    data_path = "./data/Data Set"
    print(os.listdir(data_path))
    categories = os.listdir(data_path + "/train")
    print(categories)

    for category in categories:
        if category != '.DS_Store':
            print("# Train samples for {} are:".format(category), len(os.listdir(data_path + "/train/" + category)))

    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    training_Data_Set = ImageFolder(data_path + "/train", transform=transforms)

    image, label = training_Data_Set[0]
    print(image.shape, label)
    categories = training_Data_Set.classes
    print(categories)


    def show_image(img, lab):
        print('Label: {}'.format(lab), training_Data_Set.classes[lab])
        plt.imshow(img.permute(1, 2, 0))


    show_image(*training_Data_Set[100])

    random_seed = 48
    torch.manual_seed(random_seed)

    test_size = int(len(training_Data_Set) * 0.25)
    train_size = len(training_Data_Set) - test_size
    train_data_set, test_data_set = random_split(training_Data_Set, [train_size, test_size])
    print(len(train_data_set), len(test_data_set))

    batch_size = 128

    train_data_loader = DataLoader(train_data_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data_loader = DataLoader(test_data_set, batch_size * 2, num_workers=4, pin_memory=True)


    def show_batch(dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break


    show_batch(train_data_loader)


    class Face_Mask_CNN(nn.Module):

        def __init__(self, num_classes=3):
            super(Face_Mask_CNN, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

            self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)

            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)

            self.conv4 = nn.Conv2d(in_channels=28, out_channels=36, kernel_size=3, stride=1, padding=1)

            self.pool2 = nn.MaxPool2d(kernel_size=2)

            self.conv5 = nn.Conv2d(in_channels=36, out_channels=40, kernel_size=3, stride=1, padding=1)

            self.conv6 = nn.Conv2d(in_channels=40, out_channels=48, kernel_size=3, stride=1, padding=1)

            self.pool3 = nn.MaxPool2d(kernel_size=2)

            self.fc1 = nn.Linear(in_features=16 * 16 * 48, out_features=64)

            self.fc2 = nn.Linear(in_features=64, out_features=32)

            self.fc3 = nn.Linear(in_features=32, out_features=num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))

            x = F.relu(self.pool1(self.conv2(x)))

            x = F.relu(self.conv3(x))

            x = F.relu(self.pool2(self.conv4(x)))

            x = F.relu(self.conv5(x))

            x = F.relu(self.pool3(self.conv6(x)))

            x = x.view(-1, 16 * 16 * 48)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

            return torch.log_softmax(x, dim=1)


    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model1 = Face_Mask_CNN(num_classes=len(categories)).to(device)

    print(model1)


    class Face_Mask_CNN2(nn.Module):

        def __init__(self, num_classes=3):
            super(Face_Mask_CNN2, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)

            self.conv3 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)

            self.conv4 = nn.Conv2d(in_channels=28, out_channels=36, kernel_size=3, stride=1, padding=1)

            self.conv5 = nn.Conv2d(in_channels=36, out_channels=40, kernel_size=3, stride=1, padding=1)

            self.conv6 = nn.Conv2d(in_channels=40, out_channels=48, kernel_size=3, stride=1, padding=1)

            self.fc1 = nn.Linear(in_features=64 * 64 * 48, out_features=num_classes)

        def forward(self, x):
            x = F.relu(self.pool1(self.conv1(x)))

            x = F.relu(self.conv2(x))

            x = F.relu(self.conv3(x))

            x = F.relu(self.conv4(x))

            x = F.relu(self.conv5(x))

            x = F.relu(self.conv6(x))

            x = x.view(-1, 64 * 64 * 48)

            x = self.fc1(x)

            return torch.log_softmax(x, dim=1)


    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model2 = Face_Mask_CNN2(num_classes=len(categories)).to(device)

    print(model2)


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


    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model3 = Face_Mask_CNN3(num_classes=len(categories)).to(device)

    print(model3)


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        print("Epoch:", epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = loss_criteria(output, target)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))

        avg_loss = train_loss / (batch_idx + 1)
        print('Training set: Average loss: {:.6f}'.format(avg_loss))
        return avg_loss


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            batch_count = 0
            for data, target in test_loader:
                batch_count += 1
                data, target = data.to(device), target.to(device)

                output = model(data)

                test_loss += loss_criteria(output, target).item()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(target == predicted).item()

        avg_loss = test_loss / batch_count
        print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return avg_loss


    optimizer = optim.Adam(model3.parameters(), lr=0.00001, weight_decay=0.0001)

    loss_criteria = nn.CrossEntropyLoss()

    epoch_nums = []
    training_loss = []
    validation_loss = []

    epochs = 40
    print('Training on', device)
    for epoch in range(1, epochs + 1):
        train_loss = train(model3, device, train_data_loader, optimizer, epoch)
        test_loss = test(model3, device, test_data_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

    plt.figure(figsize=(15, 15))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('./Results/loss_plot_' + model3.__class__.__name__ + '_.pdf')
    plt.show()

    truelabels = []
    predictions = []
    model3.eval().to('cpu')
    print("Getting predictions from test set...")
    for data, target in test_data_loader:
        for label in target.data.numpy():
            truelabels.append(label)
        for prediction in model3(data).data.numpy().argmax(1):
            predictions.append(prediction)

        # Plot the confusion matrix
    cm = confusion_matrix(truelabels, predictions)
    tick_marks = np.arange(len(categories))

    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig('./Results/confusion_matrix_' + model3.__class__.__name__ + '.pdf')
    plt.show()

    report = classification_report(truelabels, predictions, target_names=categories, output_dict=True)
    print(report)
    df = pd.DataFrame(report).transpose()
    df.to_csv('./Results/Classification_Report_' + model3.__class__.__name__ + '.csv')

    torch.save(model3.state_dict(), 'saved_model.pth')
