from tools.utils import *
from .dataset import SimpleDataset

class Data:
    def __init__(self, args):
        setup_seed(args.seed)
        mid_dir = creat_check_path(args.result_path, 'processed', args.dataname)
        name = "processed_{}_{}_{}.pkl".format(args.seed, args.known_cls_ratio, args.labeled_ratio)
        self.res_path = os.path.join(mid_dir, name)
        args.logger.info('load processed data: {}'.format(self.res_path))
        feats, others = load_pickle(self.res_path)
        self.train_feat, self.unlabel_train_feat, self.eval_feat, self.test_feat = feats
        self.test_labels, self.known_label_list, self.all_label_list = others
        print(self.known_label_list)
        self.num_labels = len(self.known_label_list)
        self.unseen_token = 'UNK'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        self.train_dataloader = self.get_loader(self.train_feat, args, 'train')
        self.eval_dataloader = self.get_loader(self.eval_feat, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_feat, args, 'test')
        

        
    def get_loader(self, features, args, mode = 'train'):   
        datatensor = SimpleDataset(features)
        if mode == 'train':
            dataloader = data.DataLoader(datatensor, batch_size = args.train_batch_size, shuffle=True)    
        elif mode == 'eval' or mode == 'test':
            dataloader = data.DataLoader(datatensor, batch_size = args.eval_batch_size, shuffle=False)
        return dataloader