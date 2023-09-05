from relation_discover.utils import *

class SemiOREDataset(data.Dataset):
    def __init__(self, path, tokenizer, opt=None, train=False, isunlabel=False):
        super(SemiOREDataset, self).__init__()
        
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.tokenizer = tokenizer
        self.dataname = path.split('/')[-1].split('.')[0]
        self.json_data = json.load(open(path))
        # if train:
        #     pass
        self.index = [x for x in range(len(self.json_data))]
        self.labels = [x['relid']-2 for x in self.json_data]
        self.relation_name = [x['relation'] for x in self.json_data]
        self.unique_rel_names = []
        for x in self.relation_name:
            if x not in self.unique_rel_names:
                self.unique_rel_names.append(x)
        self.gt = [x['relid'] for x in self.json_data]
        self.label_counts = {}
        self.lists_for_single_rel_mention = {}
        if opt.rel_nums < 64 and train:
            logger.info("sample part {} classes".format(opt.rel_nums))
            select_class = self.unique_rel_names[:opt.rel_nums]
        else:
            select_class = self.unique_rel_names

        for gl, item in enumerate(self.json_data):
            if item['relation'] not in select_class:
                continue
            if item["relid"] in self.label_counts:
                self.label_counts[item["relid"]] += 1
                self.lists_for_single_rel_mention[item["relid"]].append(gl)
            else:
                self.label_counts[item["relid"]] = 1
                self.lists_for_single_rel_mention[item["relid"]] = [gl]
        self.batch_size = opt.batch_size
        self.max_len = opt.max_length
        self.is_train = train
        self.is_unlabel = isunlabel
        self.use_bert = True
        self.class_num_ratio = opt.class_num_ratio

    def __getitem__(self, index):
        if self.is_train:
            batch_data, batch_label = self.get_train_data()
        else:
            batch_data, batch_label = self.get_eval_data(self.json_data[index])
            if self.is_unlabel:
                batch_label = index
        return batch_data, batch_label
    def add_bert(self, item, data_to_cluster=None):
        if data_to_cluster is None:
            data_to_cluster = {'word':[], 'pos':[], 'mask':[]}
        word, mask, pos = self.tokenizer.__getraw__(item)
        word = torch.tensor(word).long()
        pos = torch.tensor(pos).long()
        mask = torch.tensor(mask).long()
        self.tokenizer.__additem__(data_to_cluster, word, pos=pos, mask=mask)
        return data_to_cluster

    def get_eval_data(self, item):
        data_relid = []
        if self.use_bert:
            data_to_cluster = self.add_bert(item)
        else:
            data_to_cluster = self.add_cnn(item)
        return data_to_cluster, item['relid']
        
    def get_train_data(self):
        if self.use_bert:
            batch_data = {'word':[], 'pos':[], 'mask':[]}
        else:
            batch_data = {'word':[], 'pos1':[], 'pos2':[], 'pos':[], 'mask':[]}
        batch_label = []
        have_chose = dict()
        rel_chose_num = dict()
        max_class_num = len(self.lists_for_single_rel_mention.keys())
        class_num = int(self.batch_size * self.class_num_ratio)
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
                    try:
                        if self.use_bert:
                            batch_data = self.add_bert(self.json_data[instance_index], batch_data)
                        else:
                            batch_data = self.add_cnn(self.json_data[instance_index], batch_data)
                    except:
                        if len(self.json_data[instance_index]['sentence']) > self.max_len:
                            continue
                        else:
                            raise Exception("data process error")

                    have_chose[instance_index] = 1
                    rel_chose_num[index] = rel_chose_num.get(index, 0) + 1
        
                    break
        batch_label = class_list[:num]
        return batch_data, batch_label
    
    def _part_data_(self, each_class_datanum):
        data_relid = []
        data_to_cluster = []
        reltype_counter = dict()
        for item in self.json_data:
            if reltype_counter.get(item['relid'], 0) >= each_class_datanum:
                continue
            else:
                single_data, label = self.get_eval_data(item)
                data_to_cluster.append(single_data)
                data_relid.append(label)
                reltype_counter[item['relid']] = reltype_counter.get(item['relid'], 0) + 1
        return data_to_cluster, data_relid
    def __len__(self):
        if self.is_train:
            return 10000000
        else:
            return len(self.json_data)
    @staticmethod
    def coffn(data):
        
        batch_support, labels = zip(*data)
        if 'pos1' in list(batch_support[0].keys()):
            batch_data = {'word':[], 'pos1':[], 'pos2':[], 'pos':[], 'mask':[]}
        else:
            batch_data = {'word':[], 'pos':[], 'mask':[]}
        batch_label = []
        for i in range(len(batch_support)):
            for k in batch_support[i]:
                batch_data[k] += batch_support[i][k]
            if isinstance(labels, list):
                batch_label += labels[i]
            elif isinstance(labels, tuple):
                if isinstance(labels[i], int):
                    batch_label += [labels[i]]
                else:
                    batch_label += labels[i]
            else:
                batch_label += [labels[i]]
        for k in batch_data:
            batch_data[k] = torch.stack(batch_data[k], 0)
        batch_label = torch.tensor(batch_label)
        return batch_data, batch_label

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label