from tools.utils import *
from .dataset import *

class Data:
    def __init__(self, args):
        self.logger = args.logger
        setup_seed(args.seed)
        self.processor = DatasetProcessor(args)
        mid_dir = creat_check_path(args.result_path, 'processed', args.dataname)
        name = "processed_{}_{}_{}.pkl".format(args.seed, args.known_cls_ratio, args.labeled_ratio)
        self.res_path = os.path.join(mid_dir, name)
        args.logger.info('load processed data: {}'.format(self.res_path))
        feats, others = load_pickle(self.res_path)
        self.train_feat, self.unlabel_train_feat, self.eval_feat, self.test_feat = feats
        self.test_labels, self.known_label_list, self.all_label_list = others
        print(self.known_label_list)
        
        args.label_rel_nums = len(self.known_label_list)
        self.label_list = self.known_label_list
        self.unseen_token = 'UNK'
        self.unseen_token_id = args.label_rel_nums
        self.need_ind = args.need_ind
        self.get_test_labels()
        args.n_clusters = self.n_clusters
        args.num_labels = self.n_clusters


        self.data_path = os.path.join(args.data_dir, args.dataname)
        
        self.test_labels = []
        

        if args.method_type in ['semi_supervised']:
            self._sup_train_dataloader = self.get_loader(self.train_feat, args, 'train')
            self.train_dataloader = self.get_loader(self.unlabel_train_feat, args, 'train')
            self.semi_feat = self.train_feat + self.unlabel_train_feat
            self.semi_train_dataloader = self.get_loader(self.semi_feat, args, 'train')
            if args.use_sample:
                this_data = self.semi_train_dataloader.data
            else:
                this_data = self.semi_train_dataloader.dataset.data
            self.input_ids, self.batch_mask, self.batch_pos = self.transform(this_data)
        else:
            self.known_label_list = []
            self.semi_feat = self.train_feat + self.unlabel_train_feat
            self.train_dataloader = self.get_loader(self.semi_feat, args, 'train')
            
                
        self.eval_dataloader = self.get_loader(self.eval_feat, args, 'eval')
        self.test_dataloader = self.get_loader(self.test_feat, args, 'test')

        if args.method_type in ['semi_supervised']:
            if args.use_sample:
                self.sup_train_dataloader = iter(self._sup_train_dataloader)
                self.train_dataloader = iter(self.train_dataloader)
            else:
                self.sup_train_dataloader = self._sup_train_dataloader
    def get_test_labels(self):
        label_set = self.known_label_list

        for x in self.test_labels:
            if x not in label_set:
                label_set.append(x)
        self.ground_truth_num = len(label_set)
        label_map = {v:k for k,v in enumerate(label_set)}
        labels = [label_map.get(x, -1) for x in self.test_labels]
        self.test_label_ids = labels

        self.num_labels = self.ground_truth_num

        self.n_clusters = self.ground_truth_num
        
    def get_loader(self, features, args, mode='test'):
        if len(features) == 0:
            return []
    
        if args.use_sample and mode=='train':
            sample_dataset = SampleDataloader(features, args)
            return sample_dataset
        datatensor = SimpleDataset(features, output_ind=self.need_ind)
        if mode == 'train':
            dataloader = data.DataLoader(datatensor, batch_size = args.train_batch_size, shuffle=True)    
        elif mode == 'eval' or mode == 'test':
            dataloader = data.DataLoader(datatensor, batch_size = args.eval_batch_size, shuffle=False)
        
        return dataloader
    def transform(self, data):
        words, masks, poss = [], [], []
        for index in range(len(data)):
            cur = data[index]
            word = torch.tensor(cur.input_ids)
            mask = torch.tensor(cur.input_mask)
            pos = torch.tensor(cur.pos)
            words.append(word)
            masks.append(mask)
            poss.append(pos)
        words = torch.stack(words, dim=0)
        masks = torch.stack(masks, dim=0)
        poss = torch.stack(poss, dim=0)
        return words, masks, poss
