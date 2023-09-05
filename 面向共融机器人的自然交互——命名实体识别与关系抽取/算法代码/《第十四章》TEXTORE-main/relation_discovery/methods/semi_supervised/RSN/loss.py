from tools.utils import *

NEAR_0 = 1e-10
# loss functions
def sigmoid_cross_entropy_loss(pred,labels):
    loss = F.binary_cross_entropy(pred, labels)
    return loss

def sigmoid_cond_loss(pred):
    loss = -torch.mean(pred*torch.log(pred+NEAR_0)+(1-pred)*torch.log(1-pred+NEAR_0))
    return loss