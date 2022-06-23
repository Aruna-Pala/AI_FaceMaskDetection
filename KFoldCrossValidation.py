import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader

from TrainFaceMaskCNN import TrainFaceMaskCNN
from TrainFaceMaskCNN import HyperParameters
from FaceMaskCNN import FaceMaskCNN4
from EvaluateFaceMaskCNN import EvaluateFaceMaskCNN
from sklearn.model_selection import KFold

train_Face_Mask_CNN = TrainFaceMaskCNN()
eval_FM_CNN = EvaluateFaceMaskCNN()


class KFoldCrossValidation:
    def splitAndtrain(self):
        model, device, train_data = train_Face_Mask_CNN.load_train_data()
        splits = KFold(n_splits=10, shuffle=True, random_state=42)
        foldperf = {}
        classification_report = []
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
            model = FaceMaskCNN4()
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(train_data, batch_size=HyperParameters.batch_size, sampler=train_sampler)
            test_loader = DataLoader(train_data, batch_size=HyperParameters.batch_size, sampler=test_sampler)
            res_path = 'K_Fold_Cross_Validation/Fold-' + str(fold + 1) + '/'
            history, model = train_Face_Mask_CNN.trainEpoch(model, device, train_loader, test_loader,
                                                            './Results/' + res_path)
            foldperf['fold{}'.format(fold + 1)] = history
            classification_report.append(train_Face_Mask_CNN.evaluate_model(history, model, test_loader, train_data,
                                                                            res_path))
        self.calAvgPerFold(foldperf)
        res_path = 'K_Fold_Cross_Validation/Fold-avg/'
        print(eval_FM_CNN.report_average(classification_report, model, './Results/' + res_path))


    def calAvgPerFold(self, foldperf):
        testl_f, tl_f, testa_f, ta_f = [], [], [], []
        k = 10
        for f in range(1, k + 1):
            tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
            testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

            ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
            testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

        print('Performance of {} fold cross validation'.format(k))
        print(
            "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average "
            "Test "
            "Acc: {:.2f}".format(
                np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)))


if __name__ == '__main__':
    kFoldCrossValidation = KFoldCrossValidation()
    kFoldCrossValidation.splitAndtrain()
