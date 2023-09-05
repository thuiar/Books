from tools.utils import *

class BERT(nn.Module):
    
    def __init__(self,args):

        super(BERT, self).__init__()
        self.num_labels = args.num_labels
        self.encoder = SentenceEncoder(args.bert_path)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.out_dim, args.num_labels)

    def forward(self, input_ids = None, mask = None, pos=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None):

        pooled_output = self.encoder(input_ids,mask, pos)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits