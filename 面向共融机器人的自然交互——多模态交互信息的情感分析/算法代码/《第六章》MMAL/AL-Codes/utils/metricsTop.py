import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'DEMODATASET': self.__eval_demo_regression,
            'MOSI': self.__eval_MOSI_regression,
            'MOSEI': self.__eval_MOSEI_regression,
            'SIMS': self.__eval_SIMS_regression,
            'MOSIMOSEI': self.__eval_MOSIMOSEI_regression,
        }

    def __eval_demo_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_truth, test_preds)
        f1 = f1_score(test_truth, test_preds, average='macro')

        eval_results = {
            "Accuracy": accuracy,
            "F1": f1,
        }
        return eval_results

    def __eval_MOSI_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_truth, test_preds)
        f1 = f1_score(test_truth, test_preds, average='macro')

        eval_results = {
            "Accuracy": accuracy,
            "F1": f1,
        }
        return eval_results
    
    def __eval_MOSEI_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_truth, test_preds)
        f1 = f1_score(test_truth, test_preds, average='macro')

        eval_results = {
            "Accuracy": accuracy,
            "F1": f1,
        }
        return eval_results

    def __eval_SIMS_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_truth, test_preds)
        f1 = f1_score(test_truth, test_preds, average='macro')

        eval_results = {
            "Accuracy": accuracy,
            "F1": f1,
        }
        return eval_results

    def __eval_MOSIMOSEI_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.cpu().detach().numpy()
        test_truth = y_true.cpu().detach().numpy()

        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_truth, test_preds)
        f1 = f1_score(test_truth, test_preds, average='macro')

        eval_results = {
            "Accuracy": accuracy,
            "F1": f1,
        }
        return eval_results
   
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]