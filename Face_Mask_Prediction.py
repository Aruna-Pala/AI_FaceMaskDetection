from Face_Mask_CNN import Face_Mask_CNN3
import torch
import zipfile
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

if __name__ == '__main__':
    model = Face_Mask_CNN3(num_classes=5)
    model.load_state_dict(torch.load('saved_model.pth'))
    model.eval()

    with zipfile.ZipFile("./Data_Set.zip", 'r') as zip:
        zip.extractall(path="./data")

    data_path = "./data/Data Set"
    print(os.listdir(data_path))
    categories = os.listdir(data_path + "/Prediction")
    print(categories)

    batch_size = 128

    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    prediction_Data_Set = ImageFolder(data_path + "/Prediction", transform=transforms)
    prediction_data_loader = DataLoader(prediction_Data_Set, batch_size, shuffle=True, num_workers=4, pin_memory=True)

    truelabels = []
    predictions = []
    images = []
    model.eval().to('cpu')
    pos = 1
    print("Getting predictions from Prediction set...")
    for data, target in prediction_data_loader:
        for label in target.data.numpy():
            truelabels.append(label)
        for prediction in model(data).data.numpy().argmax(1):
            predictions.append(prediction)
            pos += 1
        for img in data:
            images.append(img)
    plt.rcParams["figure.figsize"] = [12, 70]
    plt.rcParams["figure.autolayout"] = True

    print('P: Predicted class \n A: Actual class')
    for i in range(1, len(images)):
        plt.subplot(20, 6, i)
        plt.imshow(images[i - 1].permute(1, 2, 0))
        plt.title("P:{}  \n A :{}".format(prediction_Data_Set.classes[predictions[i - 1]],
                                          prediction_Data_Set.classes[truelabels[i - 1]]))
    plt.savefig('./Results/Prediction/prediction.pdf')
    plt.show()

    cm = confusion_matrix(truelabels, predictions)
    tick_marks = np.arange(len(categories))
    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel("Predicted Shape", fontsize=20)
    plt.ylabel("True Shape", fontsize=20)
    plt.savefig('./Results/Prediction/Confusion_matrix.pdf')
    plt.show()

    report = classification_report(truelabels, predictions, target_names=categories)
    print(report)
