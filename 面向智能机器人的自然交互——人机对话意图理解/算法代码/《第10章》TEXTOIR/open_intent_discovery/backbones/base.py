import os 
import torch
import math
import logging
from pytorch_pretrained_bert.optimization import BertAdam
from .utils import freeze_bert_parameters, set_allow_growth
from .__init__ import backbones_map


class ModelManager:

    def __init__(self, args, data, logger_name = 'Discovery'):
        
        self.logger = logging.getLogger(logger_name)
        
        if args.backbone.startswith('bert'):
            self.model = self.set_model(args, data, 'bert')
            self.optimizer = self.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion) 
        elif args.backbone.startswith('glove'):
            self.emb_train, self.emb_test = self.set_model(args, data, 'glove')
        elif args.backbone.startswith('sae'):
            self.sae = self.set_model(args, data, 'sae')

    def set_optimizer(self, model, num_train_examples, train_batch_size, num_train_epochs, lr, warmup_proportion):

        num_train_optimization_steps = int(num_train_examples / train_batch_size) * num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                        lr = lr,
                        warmup = warmup_proportion,
                        t_total = num_train_optimization_steps)  
        return optimizer

    def set_model(self, args, data, pattern):
        
        backbone = backbones_map[args.backbone]

        if pattern == 'bert':
            self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')   
            model = backbone.from_pretrained(args.bert_model, cache_dir = "", args = args)    

            if args.freeze_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                model = freeze_bert_parameters(model)
            
            model.to(self.device)
            
            return model

        elif args.setting == 'unsupervised':

            set_allow_growth(args.gpu_id)

            if pattern == 'glove':

                self.logger.info("Building GloVe (D=300)...")
                
                gev = backbone(data.dataloader.embedding_matrix, data.dataloader.index_word, data.dataloader.train_data)
                emb_train = gev.transform(data.dataloader.train_data, method='mean')
                emb_test = gev.transform(data.dataloader.test_data, method='mean')

                self.logger.info('Building finished!')

                return emb_train, emb_test
        
            elif pattern == 'sae':

                self.logger.info("Building TF-IDF Vectors...")
                sae = backbone(data.dataloader.tfidf_train.shape[1])

                return sae




        









