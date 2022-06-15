import torch
from torch import nn
import torch.optim as optim

from DataLoaderHelper import DataLoaderHelper
from FaceMaskCNN import FaceMaskCNN4
from EvaluateFaceMaskCNN import EvaluateFaceMaskCNN


class HyperParameters:
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 40
    batch_size = 128


class TrainFaceMaskCNN:
    loss_criteria = nn.CrossEntropyLoss()

    def train(self, model, device, train_loader, optimizer, epoch, loss_criteria):
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

    def test(self, model, device, test_loader, loss_criteria):
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

    def trainEpoch(self, model, device, train_dl, test_dl):
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=HyperParameters.learning_rate,
                               weight_decay=HyperParameters.weight_decay)
        loss_criteria = nn.CrossEntropyLoss()

        epoch_nums = []
        training_loss = []
        validation_loss = []
        print('Training on', device)
        for epoch in range(1, HyperParameters.epochs + 1):
            train_loss = self.train(model, device, train_dl, optimizer, epoch, loss_criteria)
            test_loss = self.test(model, device, test_dl, loss_criteria)
            epoch_nums.append(epoch)
            training_loss.append(train_loss)
            validation_loss.append(test_loss)
        return epoch_nums, training_loss, validation_loss, model


if __name__ == '__main__':
    train_face_mask_CNN = TrainFaceMaskCNN()
    face_mask_CNN = FaceMaskCNN4()
    data_loader_helper = DataLoaderHelper()
    evaluate_face_mask_cnn = EvaluateFaceMaskCNN()
    model = face_mask_CNN
    device = face_mask_CNN.getDevice()
    path = './Data_Set.zip'
    data_path = data_loader_helper.extractZip(path)
    train_data = data_loader_helper.createTransforms(data_path + '/Train')
    train_dl, test_dl = data_loader_helper.splitData(train_data, HyperParameters.batch_size)
    data_loader_helper.showBatch(train_dl)
    epoch_nums, training_loss, validation_loss, model = train_face_mask_CNN.trainEpoch(model, device, train_dl, test_dl)
    evaluate_face_mask_cnn.plotLossGraph(epoch_nums, training_loss, validation_loss)
    evaluate_face_mask_cnn.plotConfusionMatrix(model, test_dl, train_data.classes, './Results/')
    evaluate_face_mask_cnn.saveModel(model)
