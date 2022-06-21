from FaceMaskCNN import FaceMaskCNN4
# from Face_Mask_CNN3 import Face_Mask_CNN3
from DataLoaderHelper import *
from EvaluateFaceMaskCNN import *
import torch

data_loader_helper = DataLoaderHelper()
evaluate_face_mask_CNN = EvaluateFaceMaskCNN()


class FaceMaskPrediction:
    def loadModel(self):
        # model = Face_Mask_CNN3(num_classes=5)
        model = FaceMaskCNN4(num_classes=5)
        model.load_state_dict(torch.load('./Results/Old/saved_model.pth'))
        model.eval()
        return model

    def loadData(self, path, pred_path):
        data_path = data_loader_helper.extractZip(path)
        pred_data_set = data_loader_helper.createTransforms(data_path + pred_path)
        pred_dl = data_loader_helper.createDataLoader(pred_data_set, 128)
        return pred_data_set, pred_dl

    def evaluateModelOnPred(self, model, pred_data, pred_res_path):
        evaluate_face_mask_CNN.plotConfusionMatrix(model, pred_data[1], pred_data[0].classes, pred_res_path)


if __name__ == '__main__':
    f_m_p = FaceMaskPrediction()
    model = f_m_p.loadModel()
    path = './Data_Set.zip'
    pred_pre_path = './Results/Prediction/'
    pred_path = '/Prediction'
    pred_data = f_m_p.loadData(path, pred_path)
    pred_res_path = pred_pre_path + 'Main/'
    f_m_p.evaluateModelOnPred(model, pred_data, pred_res_path)

    # prediction on biased data set - Age Elder
    pred_bias_pre_path = pred_pre_path + 'Biased/'
    print("Biased data set - Age Elder")
    path = './Data_Set.zip'
    pred_path = '/Biased/Age/Elder'
    pred_data = f_m_p.loadData(path, pred_path)
    pred_res_path = pred_bias_pre_path + 'Age/Elder/'
    f_m_p.evaluateModelOnPred(model, pred_data, pred_res_path)

    # prediction on biased data set - Age Younger
    print("Biased data set - Age Younger")
    path = './Data_Set.zip'
    pred_path = '/Biased/Age/Younger'
    pred_data = f_m_p.loadData(path, pred_path)
    pred_res_path = pred_bias_pre_path + 'Age/Younger/'
    f_m_p.evaluateModelOnPred(model, pred_data, pred_res_path)

    # prediction on biased data set - Gender Female
    print("Biased data set - Gender Female")
    path = './Data_Set.zip'
    pred_path = '/Biased/Gender/Female'
    pred_data = f_m_p.loadData(path, pred_path)
    pred_res_path = pred_bias_pre_path + 'Gender/Female/'
    f_m_p.evaluateModelOnPred(model, pred_data, pred_res_path)

    # prediction on biased data set - Gender Male
    print("Biased data set - Gender Male")
    path = './Data_Set.zip'
    pred_path = '/Biased/Gender/Male'
    pred_data = f_m_p.loadData(path, pred_path)
    pred_res_path = pred_bias_pre_path + 'Gender/Male/'
    f_m_p.evaluateModelOnPred(model, pred_data, pred_res_path)
