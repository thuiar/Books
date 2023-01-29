import torch
import math
import torch.nn as nn
from torch.autograd.function import Function

class CMLoss(nn.Module):
    def __init__(self, lm_threshold, lambda_e2l=1.0, lambda_l2e=1.0):
        super(CMLoss, self).__init__()
        self.lm_threshold = lm_threshold
        self.lambda_l = lambda_e2l
        self.lambda_e = lambda_l2e

    def forward(self, features_l, features_e, labels_l, labels_e):
        """
        features_l: (batch_size, feature_size)
        features_e: (batch_size, feature_size)
        labels_l: (batch_size, 3)
        labels_e: (batch_size)
        """
        # labels_l = labels_l.view(features_l.size(0), -1)
        labels_e = labels_e.unsqueeze(1)
        # construct features difference matrix
        fld_diff = torch.norm(features_l.unsqueeze(1) - features_l.unsqueeze(0), dim=-1) # (b * b)
        fld_diff = fld_diff / math.sqrt(features_l.size(1))
        fer_diff = torch.norm(features_e.unsqueeze(1) - features_e.unsqueeze(0), dim=-1) # (b * b)
        fer_diff = fer_diff / math.sqrt(features_e.size(1))
        fer_labels_diff = ((labels_e - labels_e.transpose(1,0)) == 0).float() # / 7 ?
        loss_l = (fld_diff * fer_labels_diff).mean()
        loss_e = (fer_diff * fer_labels_diff).mean()

        return self.lambda_l * loss_l + self.lambda_e * loss_e