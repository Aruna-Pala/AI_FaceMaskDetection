from FaceMaskCNN import FaceMaskCNN4
from DataLoaderHelper import *
from EvaluateFaceMaskCNN import *
import torch

data_loader_helper = DataLoaderHelper()
evaluate_face_mask_CNN = EvaluateFaceMaskCNN()


class FaceMaskPrediction:
    def loadModel(self):
        model = FaceMaskCNN4(num_classes=5)
        model.load_state_dict(torch.load('saved_model.pth'))
        model.eval()
        return model

    def loadData(self, path):
        data_path = data_loader_helper.extractZip(path)
        pred_data_set = data_loader_helper.createTransforms(data_path + '/Prediction')
        pred_dl = data_loader_helper.createDataLoader(pred_data_set, 128)
        return pred_data_set, pred_dl

    def evaluateModelOnPred(self, model, pred_data):
        evaluate_face_mask_CNN.plotConfusionMatrix(model, pred_data[1], pred_data[0].classes, './Results/Prediction/')


if __name__ == '__main__':
    f_m_p = FaceMaskPrediction()
    model = f_m_p.loadModel()
    path = './Data_Set.zip'
    pred_data = f_m_p.loadData(path)
    f_m_p.evaluateModelOnPred(model, pred_data)
