"""
ASIO -- All Selector in One
"""
import torch
import torch.nn as nn

from discriminators.DEMO import DEMO 
from discriminators.MSAL import MSAL 
from discriminators.CONF import CONF
from discriminators.MARGIN import MARGIN
from discriminators.ENTROPY import ENTROPY
from discriminators.MMAL import MMAL
from discriminators.LOSSPRED import LOSSPRED
from discriminators.RANDOM import RANDOM
from discriminators.CLUSTER import CLUSTER

__all__ = ['ASIO']

class ASIO():
    def __init__(self):
        self.SELECTOR_MAP = {
            'DEMO': DEMO,
            'MSAL': MSAL,
            'CONF': CONF,
            'MARGIN': MARGIN,
            'ENTROPY': ENTROPY,
            'MMAL': MMAL,
            'LOSSPRED': LOSSPRED,
            'RANDOM':RANDOM,
            'CLUSTER':CLUSTER,
        }

    def getSelector(self, args):
        return self.SELECTOR_MAP[args.selector.upper()](args)