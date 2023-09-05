from tools.utils import *
from .dataset import DatasetProcessor, SimpleDataset

class BaseData:
    def __init__(self, args):
        setup_seed(args.seed)
        self.processor = DatasetProcessor(args)
        self.data_path = os.path.join(args.data_dir, args.dataname)
        self.labeled_relation_set = None
        self.unseen_token = 'UNK'
    
    def get_train_examples(self, known_rel_list=None, labeled_ratio=1.0):
        ori_examples = self.processor.get_examples('train')
        examples = []
        unlabel_examples = []
        self.class2nums = {}
        train_labels = np.array([example.label for example in ori_examples])
        train_labeled_ids = []
        if known_rel_list is not None:
            use_rel_list = known_rel_list
        else:
            use_rel_list = self.labeled_relation_set
        for label in use_rel_list:
            self.class2nums[label] = len(train_labels[train_labels == label])
            num = round(len(train_labels[train_labels == label]) * labeled_ratio)
            pos = list(np.where(train_labels == label)[0])                
            train_labeled_ids.extend(random.sample(pos, num))

        for idx, example in enumerate(ori_examples):
            if idx in train_labeled_ids:
                examples.append(example)
            else:
                example.label = 'UNK'
                unlabel_examples.append(example)
        return examples, unlabel_examples
    def get_eval_examples(self, known_rel_list=None):
        ori_examples = self.processor.get_examples('eval')
        examples = []
        if known_rel_list is not None:
            use_rel_list = known_rel_list
        else:
            use_rel_list = self.labeled_relation_set
        for example in ori_examples:
            if (example.label in use_rel_list):
                examples.append(example)
        return examples
    def get_test_examples(self, test_rel_list=None):
        ori_examples = self.processor.get_examples('test')
        examples = []
        self.test_labels = []
        if test_rel_list is not None:
            use_rel_list = test_rel_list
        else:
            use_rel_list = self.labeled_relation_set
        for example in ori_examples:
            self.test_labels.append(example.label)
            if example.label in use_rel_list:
                examples.append(example)
            else:
                example.label = self.unseen_token
                examples.append(example)
        return examples

    def _process_labeled_data(self, ):
        # 1. process labeled data
        self.labeled_trian_examples = None
        self.labeled_eval_examples = None
        self.labeled_test_examples = None
    def _process_unlabeled_data(self,):
        # 2. process unlabeled data
        self.unlabel_train_examples = None
        self.unlabel_eval_examples = None
        self.unlabel_test_examples = None
    def _process_semi_supervised_data(self, args):
        # 3. get semi-supervised data
        self.semi_train_examples = None
        self.semi_eval_examples = None
        self.semi_test_examples = None


        self.all_label_list = list(self.processor.rel2id.keys())
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        print(self.known_label_list)

        self.num_labels = len(self.known_label_list)
        
        
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        detection_datapath = creat_check_path(args.result_path, args.task_type, 'split_data')
        this_dataname = "{}_{}_{}_{}.pkl".format(args.dataname, args.seed, args.known_cls_ratio, args.labeled_ratio)
        detection_datapath1 = os.path.join(detection_datapath, this_dataname)
        if os.path.isfile(detection_datapath1):
            args.logger.info('load split data: {}'.format(detection_datapath1))
            self.train_examples, self.eval_examples, self.test_examples, self.test_labels, self.test_texts = load_pickle(detection_datapath1)
        else:
            self.train_examples = self.get_examples(args, 'train')
            self.eval_examples = self.get_examples(args, 'eval')
            self.test_examples = self.get_examples(args, 'test')
            self.test_texts = self.processor.convert_examples_to_text(self.test_examples)
            args.logger.info('save split data: {}'.format(detection_datapath1))
            save_pickle(detection_datapath1, [self.train_examples, self.eval_examples, self.test_examples, self.test_labels, self.test_texts])
        
        this_dataname2 = "{}_{}_{}_{}_dataloader.pkl".format(args.dataname, args.seed, args.known_cls_ratio, args.labeled_ratio)
        detection_datapath2 = os.path.join(detection_datapath, this_dataname2)
        if os.path.isfile(detection_datapath2):
            args.logger.info('load splited dataloader: {}'.format(detection_datapath2))
            self.train_dataloader, self.eval_dataloader, self.test_dataloader = load_pickle(detection_datapath2)
        else:
            self.train_dataloader = self.get_loader(self.train_examples, args, 'train')
            self.eval_dataloader = self.get_loader(self.eval_examples, args, 'eval')
            self.test_dataloader = self.get_loader(self.test_examples, args, 'test')
            args.logger.info('save splited dataloader: {}'.format(detection_datapath2))
            save_pickle(detection_datapath2, [self.train_dataloader, self.eval_dataloader, self.test_dataloader])
        

        
    def get_loader(self, examples, args, mode = 'train'):   
        features = self.processor.convert_examples_to_features(examples, self.label_list)

        datatensor = SimpleDataset(features)
        
        if mode == 'train':
            dataloader = data.DataLoader(datatensor, batch_size = args.train_batch_size, shuffle=True)    
        elif mode == 'eval' or mode == 'test':
            dataloader = data.DataLoader(datatensor, batch_size = args.eval_batch_size, shuffle=False)
        
        return dataloader

    def get_examples(self, args, mode = 'train'):
        ori_examples = self.processor.get_examples(mode)
        
        examples = []
        class2nums = {}
        if mode == 'train':
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                class2nums[label] = len(train_labels[train_labels == label])
                num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])                
                train_labeled_ids.extend(random.sample(pos, num))

            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    examples.append(example)
        
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)

        elif mode == 'test':
            for example in ori_examples:
                self.test_labels.append(example.label)
                if (example.label in self.label_list) and (example.label is not self.unseen_token):
                    examples.append(example)
                    
                else:
                    example.label = self.unseen_token
                    examples.append(example)
                    
        return examples