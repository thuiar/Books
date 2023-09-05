from relation_discover.utils import *
from .tokenizer_group import get_tokenizer
from .datasets import SemiOREDataset

class Data:
    def __init__(self, args):
        args.logger = logger
        self.data_path = args.data_path
        self.train_path = self.data_path+ '/fewrel_ori/fewrel80_train.json'
        logger.info("***********dataname:{}*************".format(args.dataname))
        if args.dataname == 'fewrel':
            
            self.test_train_path = self.data_path + '/fewrel_ori/fewrel80_test_train.json'
            self.test_test_path = self.data_path + '/fewrel_ori/fewrel80_test_test.json'   
        elif args.dataname == 'cpr':
            self.test_train_path = self.data_path + '/cpr/cpr5_test_train.json'
            self.test_test_path = self.data_path + '/cpr/cpr5_test_test.json'
        elif args.dataname == 'fewrel2.0':
            self.test_train_path = self.data_path + '/pubmed/pubmed_test_train.json'
            self.test_test_path = self.data_path + '/pubmed/pubmed_test_test.json'
        
        tokenizer = get_tokenizer(args)
        
        train_dataset = SemiOREDataset(self.train_path, tokenizer, args, train=True)
        self.train_data_loader = iter(data.dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=train_dataset.coffn
        ))
        eval_data = SemiOREDataset(self.test_train_path, tokenizer, args)._part_data_(100)
        # eval data
        self.eval_loader = [train_dataset.coffn(([a, b],)) for a, b in zip(*eval_data)]
        test_dataset = SemiOREDataset(self.test_test_path, tokenizer, args)
        # test data
        self.test_data_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=test_dataset.coffn
        )
        # unlabeled train data 
        unlabel_dataset = SemiOREDataset(self.test_train_path, tokenizer, args, isunlabel=True)
        self.unlabel_loader = data.DataLoader(
            dataset=unlabel_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=unlabel_dataset.coffn
        )