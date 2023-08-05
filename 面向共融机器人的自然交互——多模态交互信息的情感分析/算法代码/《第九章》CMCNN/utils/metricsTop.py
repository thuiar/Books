import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, args):
        self.metrics_dict = {
            'FER': self.__fer_metrics,
            'LM': self.__lm_metrics,
            'FER_LM': self.__fer_lm_metrics,
        }
        self.num_landmarks = args.num_landmarks

    def __fer_metrics(self, y_pred, y_true):
        """
        Arguments: 
            y_pred: a float tensors with shape [batch_size, num_classes]
            y_true: a float tensors with shape [batch_size]
        """
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        total_acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        # https://zhuanlan.zhihu.com/p/73558315
        cm = confusion_matrix(y_true, y_pred)
        cm_float = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        average_acc = np.mean(np.diag(cm_float))
        results = {
            'Total_Acc': total_acc,
            'Average_Acc': average_acc,
            'Macro_F1': f1,
            # 'CM': cm,
        }
        return results
    
    def __lm_metrics(self, y_pred, y_true):
        """
        Arguments:
            y_pred, y_true: a float tensors with shape [batch_size, num_landmarks, 2].
        # 68 landmarks
            left_eye: 36:42
            right_eye: 42:48
        # 37 landmarks
            left_eye: 13:19
            right_eye: 19:25
        """
        batch_size = y_pred.size(0)
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()
        norm = torch.norm(y_true-y_pred, dim=2) # (batch_size, num_landmarks)
        mean_norm = torch.mean(norm, dim=1) # (batch_size)
        if self.num_landmarks == 68:
            left_eye_center = torch.mean(y_true[:,36:42,:], dim=1)
            right_eye_center = torch.mean(y_true[:,42:48,:], dim=1)
        elif self.num_landmarks == 37:
            left_eye_center = torch.mean(y_true[:,13:19,:], dim=1)
            right_eye_center = torch.mean(y_true[:,19:25,:], dim=1)

        eye_distance = torch.norm(left_eye_center-right_eye_center, dim=1)

        NME = torch.mean(mean_norm / eye_distance).item()
        results = {
            'NME': NME,
        }
        return results
    
    def __fer_lm_metrics(self, fer_y_pred, fer_y_true, lm_y_pred, lm_y_true):
        fer_results = self.__fer_metrics(fer_y_pred, fer_y_true)
        lm_results = self.__lm_metrics(lm_y_pred, lm_y_true)
        results = {**fer_results, **lm_results}
        return results
   
    def getMetrics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]