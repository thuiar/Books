"""
AMIO -- All Model in One
"""
import torch.nn as nn
from models.FER_DCNN import *
from models.LM_DCNN import *
from models.CMCNN import *

__all__ = ['AMIO']

MODEL_MAP = {
    'FER_DCNN': FER_DCNN,
    'LM_DCNN': LM_DCNN,
    'CMCNN': CMCNN
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, x):
        return self.Model(x)