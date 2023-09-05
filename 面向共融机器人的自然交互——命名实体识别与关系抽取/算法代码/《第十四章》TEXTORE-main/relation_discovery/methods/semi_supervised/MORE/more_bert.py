from tools.utils import *
from .loss import metric_loss

class MORE_BERT(nn.Module):
    def __init__(self, opt):
        super(MORE_BERT, self).__init__()
        self.opt = opt

        self.encoder = SentenceEncoder(opt.bert_path)

        self.ml = metric_loss()

        self.FFL = nn.Linear(1536, 64)
    def forward(self, word, mask, pos):
        hidden = self.encoder.get_entites(word, mask, pos)
        features = self.FFL(hidden)
        vectors = self.norm_layer(features)
        return vectors

    def norm_layer(self, bert_input):
        encoded = self.norm(bert_input)
        return encoded
    def norm(self, encoded):
        return F.normalize(encoded, p=2, dim=1)

    def get_params(self, args):
        self.lr = args.lr
        self.lr_linear = args.lr_linear
        print('# params require grad', len(list(filter(lambda p: p.requires_grad, self.Bert_model.parameters()))))

        param = []
        param += [{'params': filter(lambda p: p.requires_grad, self.encoder.parameters()), 'lr': self.lr}]
        param += [{'params': list(self.FFL.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr_linear}]
        param += [{'params': list(self.FFL.parameters())[1], 'lr': self.lr_linear}]
        return param
