import torch
from torch import nn
import torch.optim as optim

from DataLoaderHelper import DataLoaderHelper
from FaceMaskCNN import FaceMaskCNN4
from EvaluateFaceMaskCNN import EvaluateFaceMaskCNN

face_mask_CNN = FaceMaskCNN4()
data_loader_helper = DataLoaderHelper()
evaluate_face_mask_cnn = EvaluateFaceMaskCNN()


class HyperParameters:
    learning_rate = 0.001
    weight_decay = 0.01
    epochs = 40
    batch_size = 64


class TrainFaceMaskCNN:
    loss_criteria = nn.CrossEntropyLoss()

    def load_train_data(self):
        model = face_mask_CNN
        device = face_mask_CNN.getDevice()
        path = './Data_Set.zip'
        data_path = data_loader_helper.extractZip(path)
        train_data = data_loader_helper.createTransforms(data_path + '/Train')

        return model, device, train_data

    def train(self, model, device, dataloader, loss_fn, optimizer):
        train_loss, train_correct = 0.0, 0
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss, train_correct

    def test(self, model, device, dataloader, loss_fn):
        valid_loss, val_correct = 0.0, 0
        model.eval()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            valid_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            val_correct += (predictions == labels).sum().item()

        return valid_loss, val_correct

    def trainEpoch(self, model, device, train_dl, test_dl, res_path):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=HyperParameters.learning_rate,
                               weight_decay=HyperParameters.weight_decay)
        loss_criteria = nn.CrossEntropyLoss()

        print('Training on', device)
        history = {'epoch_nums': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        num_epochs = HyperParameters.epochs
        least_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss, train_correct = self.train(model, device, train_dl, loss_criteria, optimizer)
            test_loss, test_correct = self.test(model, device, test_dl, loss_criteria)

            train_loss = train_loss / len(train_dl.sampler)
            train_acc = train_correct / len(train_dl.sampler) * 100
            test_loss = test_loss / len(test_dl.sampler)
            test_acc = test_correct / len(test_dl.sampler) * 100

            if test_loss < least_loss:
                least_loss = test_loss
                evaluate_face_mask_cnn.saveModel(model, res_path)

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {"
                ":.2f} %".format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc))
            history['epoch_nums'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        return history, model

    def train_model(self, res_path):
        model, device, train_data = self.load_train_data()
        train_dl, test_dl = data_loader_helper.splitData(train_data, HyperParameters.batch_size)
        data_loader_helper.showBatch(train_dl)
        res_path = './Results/' + res_path
        history, model = self.trainEpoch(model, device, train_dl, test_dl, res_path)
        return history, model, test_dl, train_data

    def evaluate_model(self, history, model, test_dl, train_data, res_path):
        res_path = './Results/' + res_path
        epoch_nums, training_loss, validation_loss = history['epoch_nums'], history['train_loss'], history['test_loss']
        evaluate_face_mask_cnn.plotLossGraph(epoch_nums, training_loss, validation_loss, res_path)
        return evaluate_face_mask_cnn.plotConfusionMatrix(model, test_dl, train_data.classes, res_path)
        # evaluate_face_mask_cnn.saveModel(model, res_path)


if __name__ == '__main__':
    train_face_mask_CNN = TrainFaceMaskCNN()
    res_path = 'Main/'
    history, model, test_dl, train_data = train_face_mask_CNN.train_model(res_path)
    train_face_mask_CNN.evaluate_model(history, model, test_dl, train_data, res_path)
