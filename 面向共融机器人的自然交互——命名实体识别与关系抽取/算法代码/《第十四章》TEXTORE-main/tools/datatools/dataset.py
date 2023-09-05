from multiprocessing import Pool
from torch.utils.data.dataset import TensorDataset
from tools.utils import *
from .tokenizer_group import get_tokenizer

# dataset element
class Instance(object):
    def __init__(self, guid, token, h, t, relation=None):
        self.guid = guid
        self.token = token
        self.h = h
        self.t = t
        self.label = relation

# Dataset element
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, pos, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.pos = pos
        self.label_id = label_id

class DatasetProcessor(object):

    def __init__(self, args):
        super().__init__()
        setup_seed(args.seed)
        self.logger = args.logger
        try:

            self.num_worker = args.num_worker
        except:
            self.num_worker = 16
        if args.dataname in ['wiki80', 'semeval', 'wiki20m']:
            self.sf = True
        else:
            self.sf = False
        args.sf = self.sf
        self.tokenizer = get_tokenizer(args)
        self.path = os.path.join(args.data_dir, args.dataname)
        
        self.rel2id = self.get_labels(self.path)

    def get_examples(self, mode):
        data_dir = self.path
        if mode == 'train':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "train.json")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "dev.json")), "val")
        elif mode == 'test':
            return self._create_examples(
                self._read_data(os.path.join(data_dir, "test.json")), "test")
    def get_labels(self, data_dir):
        with open(os.path.join(data_dir, "rel2id.json"), encoding='utf-8') as f:
            labels = json.load(f)
        return labels
    def _read_data(self, path:str):
        if path.endswith('txt'):
            f = open(path, encoding='utf-8')
            data = []
            for line in f.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    data.append(eval(line))
            f.close()
        elif path.endswith('json'):
            data = self.read_json(path)
        return data
    def read_json(self, path):
        self.logger.info('read data from {}'.format(path))
        data = load_json(path)
        self.logger.info('data length is {}'.format(len(data)))
        return data

    def convert_features_to_tensor(self, features):
        words = torch.stack([torch.tensor(x.input_ids) for x in features], dim=0)
        masks =torch.stack([torch.tensor(x.input_mask) for x in features], dim=0)
        pos = torch.stack([torch.tensor(x.pos) for x in features], dim=0)
        labels = torch.stack([torch.tensor(x.label_id) for x in features], dim=0)
        return words, masks, pos, labels
    def _create_examples(self, lines, set_type):
        """Creates examples for the dataset."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if 'token' in line:
                examples.append(
                    Instance(guid=guid, token=line['token'], h=line['h'], t=line['t'], relation=line['relation']))
            else:
                examples.append(
                    Instance(guid=guid, token=line['tokens'], h=line['h'], t=line['t'], relation=line['relation']))
        return examples
    def convert_examples_to_text(self, examples):
        all_data = []
        for item in examples:
            token = item.token
            text = " ".join(token)
            relation = item.label
            h = {
                "e1": item.token[item.h['pos'][0]:item.h['pos'][-1]+1],
                "pos": item.h['pos']
            }
            t = {
                "e2": item.token[item.t['pos'][0]:item.t['pos'][-1]+1],
                "pos": item.t['pos']
            }
            all_data.append(
                {"text": text, "label": relation, "h":h, "t": t}
            )
        return all_data

    def get_features(self, x):
        res = self.tokenizer.tokenize(x.token, x.h['pos'], x.t['pos'])
        label = self.label_map.get(x.label, -1)
        return InputFeatures(*res, label_id=label)
    def convert_examples_to_features(self, examples, label_list, label_map=None, only_unk = None):
        self.logger.info("convert_examples_to_features.....")
        # pool = Pool(processes=self.num_worker)
        if label_map is None:
            label_map = {}
            for i, label in enumerate(label_list):
                label_map[label] = i
        if only_unk is not None:
            label_map[label_list[0]] = -1
        features = []
        self.label_map = label_map
        # def append_feats(x):
        #     features.append(x)
        
        self.logger.info("label map.....{}".format(str(label_map)))
        self.logger.info("sample nums: {}".format(len(examples)))
        for item in examples:
            # features.append(pool.map_async(self.get_features, (item,)))
            features.append(self.get_features(item))
        # pool.close()
        # pool.join()
        # features = [x.get()[0] for x in features]
        return features
    def split_data(self, data=None):
        if data is None:
            if 'fewrel' in self.path:
                data = self.read_json(os.path.join(self.path, "dev.json"))
            elif 'pubmed' in self.path:
                data = self.read_json(os.path.join(self.path, "pubmed.json"))
            else:
                train_data = self.read_json(os.path.join(self.path, "train.json"))
                val_data = self.read_json(os.path.join(self.path, "dev.json"))
                test_data = self.read_json(os.path.join(self.path, "test.json"))
                data = train_data + val_data + test_data
        test_len = min(1600, len(data)//2)
        test_idx = np.random.choice(np.arange(len(data)), test_len, replace=False)
        new_lines_test_test = []
        new_lines_test_train = []
        for idx,item in enumerate(data):
            if "semeval" in self.path:
                item["h"]['pos'][-1] = item["h"]['pos'][-1]-1
                item["t"]['pos'][-1] = item["t"]['pos'][-1]-1
            if idx in test_idx:
                new_lines_test_test.append(item)
            else:
                new_lines_test_train.append(item)
        unsup_train_data = self._create_examples(new_lines_test_train, "unsup_train")
        unsup_test_data = self._create_examples(new_lines_test_test, "unsup_test")
        return unsup_train_data, unsup_test_data

#############################################################
#########     Simple Dataset
#############################################################

class SimpleDataset(data.Dataset):
    def __init__(self, data, output_ind = 0):
        super(SimpleDataset, self).__init__()
        self.data = data
        self.output_ind = output_ind
    def __getitem__(self, index):
        cur = self.data[index]
        word = torch.tensor(cur.input_ids)
        mask = torch.tensor(cur.input_mask)
        pos = torch.tensor(cur.pos)
        label = torch.tensor(cur.label_id)
        if self.output_ind:
            ind = torch.tensor(index)
            return word, mask, pos, ind, label

        return word, mask, pos, label
    def __len__(self):
        return len(self.data)

#############################################################
#########     Sample Dataset
#############################################################

class SampleDataloader(data.Dataset):
    """
    for metric training
    """
    def __init__(self, data, args):
        super(SampleDataloader, self).__init__()
        self.data = data
        self.label_counts = {}
        self.lists_for_single_rel_mention = {}
        self.batch_size = args.train_batch_size
        self.class_num_ratio = args.class_num_ratio
        for gl, item in enumerate(self.data):
            if item.label_id in self.label_counts:
                self.label_counts[item.label_id] += 1
                self.lists_for_single_rel_mention[item.label_id].append(gl)
            else:
                self.label_counts[item.label_id] = 1
                self.lists_for_single_rel_mention[item.label_id] = [gl]

    def __getitem__(self, index):
        return self.get_train_data()
    def __len__(self):
        return 10000000
        
    def get_train_data(self):
        batch_data = []
        batch_label = []
        have_chose = dict()
        rel_chose_num = dict()
        max_class_num = len(self.lists_for_single_rel_mention.keys())
        class_num = min(max_class_num, int(self.batch_size * self.class_num_ratio) )
        class_list = random.sample(list(self.lists_for_single_rel_mention.keys()), class_num)
        class_list = class_list * (self.batch_size // class_num + 1)
        num = self.batch_size
        for i, index in enumerate(class_list):
            if i >= num:
                break
            while True:
                instance_index = random.choice(self.lists_for_single_rel_mention[index])
                if have_chose.get(instance_index, 0) == 0 or rel_chose_num[index] >= len(
                        self.lists_for_single_rel_mention[index]):
                    batch_data.append(self.data[instance_index])
                    have_chose[instance_index] = 1
                    rel_chose_num[index] = rel_chose_num.get(index, 0) + 1
                    break
        batch_label = class_list[:num]
        words = torch.stack([torch.tensor(item.input_ids) for item in batch_data], dim=0)
        masks = torch.stack([torch.tensor(item.input_mask) for item in batch_data], dim=0)
        poss = torch.stack([torch.tensor(item.pos) for item in batch_data], dim=0)
        batch_label = torch.tensor(batch_label)
        return words, masks, poss, batch_label
    def _part_data_(self, each_class_datanum):
        # 遍历整个数据集，对每种relation采不超过each_num_class个
        # adding batch number shape
        words = []
        masks = []
        poss = []
        labels = []
        data_to_cluster = []
        reltype_counter = dict()
        for item in self.data:
            if reltype_counter.get(item.label_id, 0) >= each_class_datanum:
                continue
            else:
                word = torch.tensor(item.input_ids)
                mask = torch.tensor(item.input_mask)
                pos = torch.tensor(item.pos)
                label = torch.tensor(item.label_id)
                words.append(word)
                masks.append(mask)
                poss.append(pos)
                labels.append(label)
                reltype_counter[item.label_id] = reltype_counter.get(item.label_id, 0) + 1
        words = torch.stack(words, dim=0)
        masks = torch.stack(masks, dim=0)
        poss = torch.stack(poss, dim=0)
        labels = torch.stack(labels, dim=0)
        return words, masks, poss, labels
    @staticmethod
    def coffn(data):
        return data[0]

# from sys import path_hooks

# from torch.utils.data.dataset import TensorDataset
# from tools.utils import *
# from .tokenizer_group import get_tokenizer

# # dataset element
# class Instance(object):
#     def __init__(self, guid, token, h, t, relation=None):
#         self.guid = guid
#         self.token = token
#         self.h = h
#         self.t = t
#         self.label = relation

# # Dateset element
# class InputFeatures(object):
#     def __init__(self, input_ids, input_mask, pos, label_id):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.pos = pos
#         self.label_id = label_id

# class DatasetProcessor(object):

#     def __init__(self, args):
#         super().__init__()
#         setup_seed(args.seed)
#         self.logger = args.logger

#         if args.dataname in ['wiki80', 'semeval', 'wiki20m']:
#             self.sf = True
#         else:
#             self.sf = False
#         args.sf = self.sf
#         self.tokenizer = get_tokenizer(args)
#         self.path = os.path.join(args.data_dir, args.dataname)
        
#         self.rel2id = self.get_labels(self.path)

#     def get_examples(self, mode):
#         data_dir = self.path
#         if mode == 'train':
#             return self._create_examples(
#                 self._read_data(os.path.join(data_dir, "train.json")), "train")
#         elif mode == 'eval':
#             return self._create_examples(
#                 self._read_data(os.path.join(data_dir, "dev.json")), "val")
#         elif mode == 'test':
#             return self._create_examples(
#                 self._read_data(os.path.join(data_dir, "test.json")), "test")
#     def get_labels(self, data_dir):
#         with open(os.path.join(data_dir, "rel2id.json"), encoding='utf-8') as f:
#             labels = json.load(f)
#         return labels
#     def _read_data(self, path:str):
#         if path.endswith('txt'):
#             f = open(path, encoding='utf-8')
#             data = []
#             for line in f.readlines():
#                 line = line.rstrip()
#                 if len(line) > 0:
#                     data.append(eval(line))
#             f.close()
#         elif path.endswith('json'):
#             data = self.read_json(path)
#         return data
#     def read_json(self, path):
#         self.logger.info('read data from {}'.format(path))
#         data = load_json(path)
#         self.logger.info('data length is {}'.format(len(data)))
#         return data

#     def convert_features_to_tensor(self, features):
#         words = torch.stack([torch.tensor(x.input_ids) for x in features], dim=0)
#         masks =torch.stack([torch.tensor(x.input_mask) for x in features], dim=0)
#         pos = torch.stack([torch.tensor(x.pos) for x in features], dim=0)
#         labels = torch.stack([torch.tensor(x.label_id) for x in features], dim=0)
#         return words, masks, pos, labels
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the dataset."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             if 'token' in line:
#                 examples.append(
#                     Instance(guid=guid, token=line['token'], h=line['h'], t=line['t'], relation=line['relation']))
#             else:
#                 examples.append(
#                     Instance(guid=guid, token=line['tokens'], h=line['h'], t=line['t'], relation=line['relation']))
#         return examples
#     def convert_examples_to_text(self, examples):
#         all_data = []
#         for item in examples:
#             token = item.token
#             text = " ".join(token)
#             relation = item.label
#             h = {
#                 "e1": item.token[item.h['pos'][0]:item.h['pos'][-1]+1],
#                 "pos": item.h['pos']
#             }
#             t = {
#                 "e2": item.token[item.t['pos'][0]:item.t['pos'][-1]+1],
#                 "pos": item.t['pos']
#             }
#             all_data.append(
#                 {"text": text, "label": relation, "h":h, "t": t}
#             )
#         return all_data

#     def convert_examples_to_features(self, examples, label_list, label_map=None):
#         if label_map is None:
#             label_map = {}
#             for i, label in enumerate(label_list):
#                 label_map[label] = i
#         features = []
#         for item in examples:
#             input_ids, mask, pos = self.tokenizer.tokenize(item.token, item.h['pos'], item.t['pos'])
#             label = label_map.get(item.label, -1)
#             features.append(
#             InputFeatures(input_ids=input_ids,
#                           input_mask=mask,
#                           pos = pos,
#                           label_id=label))
#         return features
#     def split_data(self, data=None):
#         if data is None:
#             if 'fewrel' in self.path:
#                 data = self.read_json(os.path.join(self.path, "dev.json"))
#             elif 'pubmed' in self.path:
#                 data = self.read_json(os.path.join(self.path, "pubmed.json"))
#             else:
#                 train_data = self.read_json(os.path.join(self.path, "train.json"))
#                 val_data = self.read_json(os.path.join(self.path, "dev.json"))
#                 test_data = self.read_json(os.path.join(self.path, "test.json"))
#                 data = train_data + val_data + test_data
#         test_len = min(1600, len(data)//2)
#         test_idx = np.random.choice(np.arange(len(data)), test_len, replace=False)
#         new_lines_test_test = []
#         new_lines_test_train = []
#         for idx,item in enumerate(data):
#             if "semeval" in self.path:
#                 item["h"]['pos'][-1] = item["h"]['pos'][-1]-1
#                 item["t"]['pos'][-1] = item["t"]['pos'][-1]-1
#             if idx in test_idx:
#                 new_lines_test_test.append(item)
#             else:
#                 new_lines_test_train.append(item)
#         unsup_train_data = self._create_examples(new_lines_test_train, "unsup_train")
#         unsup_test_data = self._create_examples(new_lines_test_test, "unsup_test")
#         return unsup_train_data, unsup_test_data

# #############################################################
# #########     Simple Dataset
# #############################################################

# class SimpleDataset(data.Dataset):
#     def __init__(self, data, output_ind = 0):
#         super(SimpleDataset, self).__init__()
#         self.data = data
#         self.output_ind = output_ind
#     def __getitem__(self, index):
#         cur = self.data[index]
#         word = torch.tensor(cur.input_ids)
#         mask = torch.tensor(cur.input_mask)
#         pos = torch.tensor(cur.pos)
#         label = torch.tensor(cur.label_id)
#         if self.output_ind:
#             ind = torch.tensor(index)
#             return word, mask, pos, ind, label

#         return word, mask, pos, label
#     def __len__(self):
#         return len(self.data)

# #############################################################
# #########     Sample Dataset
# #############################################################

# class SampleDataloader(data.Dataset):
#     """
#     for metric training
#     """
#     def __init__(self, data, args):
#         super(SampleDataloader, self).__init__()
#         self.data = data
#         self.label_counts = {}
#         self.lists_for_single_rel_mention = {}
#         self.batch_size = args.train_batch_size
#         self.class_num_ratio = args.class_num_ratio
#         for gl, item in enumerate(self.data):
#             if item.label_id in self.label_counts:
#                 self.label_counts[item.label_id] += 1
#                 self.lists_for_single_rel_mention[item.label_id].append(gl)
#             else:
#                 self.label_counts[item.label_id] = 1
#                 self.lists_for_single_rel_mention[item.label_id] = [gl]

#     def __getitem__(self, index):
#         return self.get_train_data()
#     def __len__(self):
#         return 10000000
        
#     def get_train_data(self):
#         batch_data = []
#         batch_label = []
#         have_chose = dict()
#         rel_chose_num = dict()
#         max_class_num = len(self.lists_for_single_rel_mention.keys())
#         class_num = min(max_class_num, int(self.batch_size * self.class_num_ratio) )
#         class_list = random.sample(list(self.lists_for_single_rel_mention.keys()), class_num)
#         class_list = class_list * (self.batch_size // class_num + 1)
#         num = self.batch_size
#         for i, index in enumerate(class_list):
#             if i >= num:
#                 break
#             while True:
#                 instance_index = random.choice(self.lists_for_single_rel_mention[index])
#                 if have_chose.get(instance_index, 0) == 0 or rel_chose_num[index] >= len(
#                         self.lists_for_single_rel_mention[index]):
#                     batch_data.append(self.data[instance_index])
#                     have_chose[instance_index] = 1
#                     rel_chose_num[index] = rel_chose_num.get(index, 0) + 1
#                     break
#         batch_label = class_list[:num]
#         words = torch.stack([torch.tensor(item.input_ids) for item in batch_data], dim=0)
#         masks = torch.stack([torch.tensor(item.input_mask) for item in batch_data], dim=0)
#         poss = torch.stack([torch.tensor(item.pos) for item in batch_data], dim=0)
#         batch_label = torch.tensor(batch_label)
#         return words, masks, poss, batch_label
#     def _part_data_(self, each_class_datanum):
#         # 遍历整个数据集，对每种relation采不超过each_num_class个
#         # adding batch number shape
#         words = []
#         masks = []
#         poss = []
#         labels = []
#         data_to_cluster = []
#         reltype_counter = dict()
#         for item in self.data:
#             if reltype_counter.get(item.label_id, 0) >= each_class_datanum:
#                 continue
#             else:
#                 word = torch.tensor(item.input_ids)
#                 mask = torch.tensor(item.input_mask)
#                 pos = torch.tensor(item.pos)
#                 label = torch.tensor(item.label_id)
#                 words.append(word)
#                 masks.append(mask)
#                 poss.append(pos)
#                 labels.append(label)
#                 reltype_counter[item.label_id] = reltype_counter.get(item.label_id, 0) + 1
#         words = torch.stack(words, dim=0)
#         masks = torch.stack(masks, dim=0)
#         poss = torch.stack(poss, dim=0)
#         labels = torch.stack(labels, dim=0)
#         return words, masks, poss, labels
#     @staticmethod
#     def coffn(data):
#         return data[0]
