from tools.datatools.tokenizer_group import *
from tools.datatools.dataset import DatasetProcessor


class Data:
    def __init__(self, args):
        setup_seed(args.seed)
        self.logger = args.logger
        self.processor = DatasetProcessor(args)
        self.data_path = os.path.join(args.data_dir, args.dataname)
        self.all_label_list = list(self.processor.rel2id.keys())
        if "Other" in self.all_label_list:
            self.all_label_list.remove("Other")
        if "Entity-Destination(e2,e1)" in self.all_label_list:
            self.all_label_list.remove("Entity-Destination(e2,e1)")
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        print(self.known_label_list)

        self.num_labels = len(self.known_label_list)
        self.unseen_token = 'UNK'
        self.test_labels = []
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]
        
        self.train_examples, self.unlabel_examples = self.get_examples(args, 'train')

        self.eval_examples = self.get_examples(args, 'eval')
        self.test_examples = self.get_examples(args, 'test')
        self.logger.info("get train_feat.....")
        self.train_feat = self.get_feat(self.train_examples)
        self.logger.info("get unlabel_train_feat.....")
        self.unlabel_train_feat = self.get_feat(self.unlabel_examples)
        self.logger.info("get eval_feat.....")
        self.eval_feat = self.get_feat(self.eval_examples)
        self.logger.info("get test_feat.....")
        self.test_feat = self.get_feat(self.test_examples)

        save_pickle(args.res_path, 
            [[self.train_feat, self.unlabel_train_feat, self.eval_feat, self.test_feat], 
            [self.test_labels,
            self.known_label_list,
            self.all_label_list,
            ]])
        
    def get_feat(self, examples, unlabel=False):
        features = self.processor.convert_examples_to_features(examples, self.label_list)
        return features

    def get_examples(self, args, mode = 'train'):
        ori_examples = self.processor.get_examples(mode)
        
        examples = []
        class2nums = {}
        if mode == 'train':
            unlabel_examples = []
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
                else:
                    example.label = self.unseen_token
                    unlabel_examples.append(example)
            return examples, unlabel_examples
        
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