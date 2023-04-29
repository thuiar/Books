"""BERT finetuning runner."""
from __future__ import absolute_import, division, print_function
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertConfig,BertModel, WEIGHTS_NAME, CONFIG_NAME
from torch.nn.utils import weight_norm
from losses import *
from torch.nn.parameter import Parameter
import torch
import math
###################MSP##################3
class BertForMSP(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForMSP, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits,labels)
                return logits, loss
            else:
                return pooled_output, logits
            
###############DOC####################
class BertForDOC(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForDOC, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if feature_ext:
            return pooled_output,logits
        else:
            return pooled_output, logits
    
        
#######################OSDN######################3        
class BertForOSDN(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForOSDN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
#         pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim = 1)
        
        if feature_ext == True:
            return logits, preds, probs
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits,labels)
                return logits, loss
            else:        
                return pooled_output, logits
            
#########ArcFace#####################    
class BertForIntentDetect_ArcFace(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForIntentDetect_ArcFace,self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = L2_normalization()
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        ##self.classifier = weight_norm(nn.Linear(config.hidden_size, num_labels,bias=False),name='weight')
        self.classifier.weight = torch.nn.Parameter(self.classifier.weight / torch.norm(self.classifier.weight, dim=1, keepdim=True))
        self.apply(self.init_bert_weights)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, get_feature=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        feature = pooled_output
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = AMSoftmaxLoss(margin_type='arc',s=64,m=0.35)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
            return loss
        elif get_feature:
            return feature
        else:
            return logits 
        
######################NormFace#####################
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores    
class BertForIntentDetect_NormFace(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForIntentDetect_NormFace,self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = My_l2_norm()
        self.classifier = distLinear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, get_feature=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        feature = pooled_output
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
            return loss
        elif get_feature:
            return feature
        else:
            return logits
##################LMCL (AM-Softmax)###################
class BertForIntentDetect(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForIntentDetect,self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm = L2_normalization()
        self.classifier = weight_norm(nn.Linear(config.hidden_size, num_labels),name='weight')
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, get_feature=False):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.norm(pooled_output)
        feature = pooled_output
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = AMSoftmaxLoss(m=0.35)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif get_feature:
            return feature
        else:
            return feature, logits 
        
###################Meta-Embedding##############
class BertForMetaEmbedding(BertPreTrainedModel):
    def __init__(self,config,num_labels):
        super(BertForMetaEmbedding, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.fc_hallucinator = nn.Linear(config.hidden_size, num_labels)
        self.fc_selector = nn.Linear(config.hidden_size, config.hidden_size)
        self.cosnorm_classifier = CosNorm_Classifier(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, centroids = None, loss_name = None):

        encoded_layer_12, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers = True)
        pooled_output = self.dense(encoded_layer_12[-1].mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        x = pooled_output

        if feature_ext:
            return x
        else:
            direct_feature = x
            batch_size = x.size(0)
            feat_size = x.size(1)

            #set up visual memory
            x_expand = x.unsqueeze(1).expand(-1, self.num_labels, -1)
            centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
            keys_memory = centroids

            #computing reachability
            dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
            # if phase == 'test':
            #     print(dist_cur)
            #     print(torch.mean(dist_cur,1))
            values_nn, labels_nn = torch.sort(dist_cur, 1)
            #dis = torch.mean(dist_cur,1)
            dis = values_nn[:, 0]
            scale = 1.0
            reachability = (scale / dis).unsqueeze(1).expand(-1, feat_size)
            x = reachability * direct_feature
            logits = self.cosnorm_classifier(x)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss_ce = loss_fct(logits.view(-1, self.num_labels), labels.view(-1) )
                if loss_name == 'softmax':
                    loss = loss_ce
                else:
                    if loss_name == 'AM-softmax':
                        loss_fct = AMSoftmaxLoss(in_feats = feat_size)
                        loss_feat = loss_fct(direct_feature, labels)
                    elif loss_name == 'Arc-Face':
                        loss_fct = ArcFaceLoss(in_features = feat_size, out_features = self.num_labels)
                        loss_feat = loss_fct(direct_feature, labels)
                    elif loss_name == 'Discentroid':
                        loss_fct = DiscCentroidsLoss(num_classes = self.num_labels, feat_dim = feat_size)
                        loss_feat = loss_fct(direct_feature, labels, centroids)
                    loss = loss_ce + 0.1 * loss_feat
                    
                return loss
            else:
                return logits, x
        
class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16
                 , margin=0.5, init_std=0.001):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())
    
    

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
class L2_normalization(nn.Module):
    def forward(self, input):
        return l2_norm(input)