from tools.utils import *

class OpenMax(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        logging.info('Loading BERT pre-trained checkpoint.')
        self.encoder = SentenceEncoder(args.bert_path)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.encoder.out_dim, num_labels)
    def forward(self, token, att_mask, pos, labels = None, feature_ext = False, mode=None, loss_fct=None):
        x = self.encoder(token, att_mask, pos)
        if feature_ext:
            return x
        else:
            logits = self.classifier(x)
            if mode == 'train':
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
            else:
                return x, logits