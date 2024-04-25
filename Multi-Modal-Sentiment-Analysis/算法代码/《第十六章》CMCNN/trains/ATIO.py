"""
ATIO -- All Train In One
"""
from trains.FER_DCNN import *
from trains.LM_DCNN import *
from trains.CMCNN import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'FER_DCNN': FER_DCNN,
            'LM_DCNN': LM_DCNN,
            'CMCNN': CMCNN
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName](args)