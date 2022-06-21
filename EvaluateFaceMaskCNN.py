import os

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class EvaluateFaceMaskCNN:
    def createFolderIfNotExists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def plotLossGraph(self, epoch_nums, training_loss, validation_loss, res_path):
        plt.figure(figsize=(15, 15))
        plt.plot(epoch_nums, training_loss)
        plt.plot(epoch_nums, validation_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['training', 'validation'], loc='upper right')
        self.createFolderIfNotExists(res_path)
        plt.savefig(res_path + 'loss_plot.pdf')
        # plt.show()

    def printWrongPredictions(self, images, predictions, true_labels, categories, file_path):
        plt.rcParams["figure.figsize"] = [12, 70]
        plt.rcParams["figure.autolayout"] = True
        count = 1
        for i in range(1, len(images)):
            if predictions[i - 1] != true_labels[i - 1]:
                plt.subplot(40, 6, count)
                count += 1
                plt.imshow(images[i - 1].permute(1, 2, 0))
                plt.title("P:{}  \n A :{}".format(categories[predictions[i - 1]], categories[true_labels[i - 1]]))
            self.createFolderIfNotExists(file_path)
            plt.savefig(file_path + 'prediction.pdf')

    def printClassificationReport(self, model, predictions, truelabels, categories, file_path):
        report = classification_report(truelabels, predictions, target_names=categories, output_dict=True)
        print(report)
        df = pd.DataFrame(report).transpose()
        df.to_csv(file_path + 'Classification_Report_' + model.__class__.__name__ + '.csv')
        return report

    def report_average(self, reports, model, file_path):
        mean_dict = dict()
        for label in reports[0].keys():
            dictionary = dict()

            if label in 'accuracy':
                mean_dict[label] = sum(d[label] for d in reports) / len(reports)
                continue

            for key in reports[0][label].keys():
                dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
            mean_dict[label] = dictionary

            df = pd.DataFrame(mean_dict).transpose()
            self.createFolderIfNotExists(file_path)
            df.to_csv(file_path + 'Classification_Report_avg_' + model.__class__.__name__ + '.csv')

        return mean_dict

    def plotConfusionMatrix(self, model, test_data_loader, categories, file_path):
        true_labels = []
        predictions = []
        images = []
        model.eval().to('cpu')
        print("Getting predictions from data set...")
        for data, target in test_data_loader:
            for label in target.data.numpy():
                true_labels.append(label)
            for prediction in model(data).data.numpy().argmax(1):
                predictions.append(prediction)
            for img in data:
                images.append(img)

            # Plot the confusion matrix
        cm = confusion_matrix(true_labels, predictions)

        df_cm = pd.DataFrame(cm, index=categories, columns=categories)
        plt.figure(figsize=(7, 7))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel("Predicted Shape", fontsize=20)
        plt.ylabel("True Shape", fontsize=20)
        self.createFolderIfNotExists(file_path)
        plt.savefig(file_path + 'confusion_matrix_' + model.__class__.__name__ + '.pdf')
        # plt.show()
        self.printWrongPredictions(images, predictions, true_labels, categories, file_path)
        return self.printClassificationReport(model, predictions, true_labels, categories, file_path)

    def saveModel(self, model, path):
        self.createFolderIfNotExists(path)
        torch.save(model.state_dict(), path + 'saved_model.pth')
