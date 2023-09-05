from tools.utils import *
from torch.nn.utils import weight_norm

class CosineFaceLoss(nn.Module):

    """
    cos_theta and target need to be normalized first
    """ 

    def __init__(self, m=0.35, s=30):
        
        super(CosineFaceLoss, self).__init__()
        self.m = m
        self.s = s

    def forward(self, cos_theta, target):
        
        phi_theta = cos_theta - self.m

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)
        
        return F.cross_entropy(self.s * output, target)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class L2_normalization(nn.Module):
    def forward(self, input):
        return l2_norm(input)

class DeepUnk(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        logging.info('Loading BERT pre-trained checkpoint.')
        self.encoder = SentenceEncoder(args.bert_path)
        self.num_labels = num_labels
        self.norm = L2_normalization()
        self.classifier = weight_norm(
            nn.Linear(self.encoder.out_dim, num_labels), name='weight')
    def forward(self, token, att_mask, pos, labels = None, feature_ext = False, mode=None, loss_fct=None):
        x = self.encoder(token, att_mask, pos)
        x = self.norm(x)
        if feature_ext:
            return x
        else:
            logits = self.classifier(x)
            if mode == 'train':
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
            else:
                return x, logits