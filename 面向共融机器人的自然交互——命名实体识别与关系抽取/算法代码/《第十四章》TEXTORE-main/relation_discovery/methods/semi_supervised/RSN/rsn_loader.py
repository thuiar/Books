from torch.utils.data.dataset import TensorDataset
from tools.utils import *
class RSNDataloader(data.Dataset):

    def __init__(self, data, args):
        super(RSNDataloader, self).__init__()
        self.data = data
        self.label_counts = {}
        self.lists_for_single_rel_mention = {}
        self.batch_size = args.batch_size
        for gl, item in enumerate(self.data):
            if item.label_id in self.label_counts:
                self.label_counts[item.label_id] += 1
                self.lists_for_single_rel_mention[item.label_id].append(gl)
            else:
                self.label_counts[item.label_id] = 1
                self.lists_for_single_rel_mention[item.label_id] = [gl]

    def next_batch_same(self, batch_size): # return a list
        batch_data_same_left = []
        batch_data_same_right = []
        for i in range(batch_size):
            next_rel_index = random.choice(list(self.lists_for_single_rel_mention.keys()))
            if len(self.lists_for_single_rel_mention[next_rel_index])==1:
                temp_index = self.lists_for_single_rel_mention[next_rel_index] * 2
            else:
                temp_index = random.sample(self.lists_for_single_rel_mention[next_rel_index], 2)
            batch_data_same_left.append(self.data[temp_index[0]])
            batch_data_same_right.append(self.data[temp_index[1]])

        return batch_data_same_left, batch_data_same_right
    
    def next_batch_rand(self, batch_size, active_selector = None, select_num = 1): # return a list
        batch_data_rand_left = []
        batch_data_rand_right = []

        idx_list = np.arange(len(self.data))
        rnd_list = np.random.choice(idx_list, 2*batch_size*select_num)

        for i in range(batch_size*select_num):
            temp_index = rnd_list[2*i:2*i+2]
            while 1:
                if(self.data[temp_index[0]].label_id != self.data[temp_index[1]].label_id):
                    batch_data_rand_left.append(self.data[temp_index[0]])
                    batch_data_rand_right.append(self.data[temp_index[1]])
                    break
                else:
                    temp_index = np.random.choice(idx_list, 2 , replace=False)

        if active_selector is not None:
            selected_index = active_selector(batch_data_rand_left, batch_data_rand_right).argsort()[:batch_size]
            batch_data_rand_left_selected = []
            batch_data_rand_right_selected = []
            for temp_index in selected_index:
                batch_data_rand_left_selected.append(batch_data_rand_left[int(temp_index)])
                batch_data_rand_right_selected.append(batch_data_rand_right[int(temp_index)])
            batch_data_rand_left = batch_data_rand_left_selected
            batch_data_rand_right = batch_data_rand_right_selected

        return  batch_data_rand_left, batch_data_rand_right
    
    def next_batch(self, batch_size, same_ratio = 0.5, active_selector = None, select_num = 1):
        same_batch_size = int(np.round(batch_size*same_ratio))
        batch_data_same_left, batch_data_same_right = self.next_batch_same(same_batch_size)
        batch_data_rand_left, batch_data_rand_right = self.next_batch_rand(
            batch_size - same_batch_size, active_selector = active_selector, select_num = select_num)
        batch_data_left = batch_data_same_left + batch_data_rand_left
        batch_data_right = batch_data_same_right + batch_data_rand_right
        batch_data_label = [0]*len(batch_data_same_left)+[1]*len(batch_data_rand_left)

        batch_data_left_word = torch.stack([torch.tensor(x.input_ids) for x in batch_data_left], dim=0).long()
        batch_data_left_mask = torch.stack([torch.tensor(x.input_mask) for x in batch_data_left], dim=0).long()
        batch_data_left_pos = torch.stack([torch.tensor(x.pos) for x in batch_data_left], dim=0).long()

        batch_data_right_word = torch.stack([torch.tensor(x.input_ids) for x in batch_data_right], dim=0).long()
        batch_data_right_mask = torch.stack([torch.tensor(x.input_mask) for x in batch_data_right], dim=0).long()
        batch_data_right_pos = torch.stack([torch.tensor(x.pos) for x in batch_data_right], dim=0).long()

        return (batch_data_left_word, batch_data_left_mask, batch_data_left_pos), \
            (batch_data_right_word, batch_data_right_mask, batch_data_right_pos), torch.tensor(batch_data_label).float()

    def next_batch_ul(self, batch_size):
        batch_data_left = []
        batch_data_right = []
        for _ in range(batch_size):
            temp_index = random.sample(np.arange(len(self.data)).tolist(),2)
            batch_data_left.append(self.data[temp_index[0]])
            batch_data_right.append(self.data[temp_index[1]])
        
        batch_data_left_word = torch.stack([torch.tensor(x.input_ids) for x in batch_data_left], dim=0).long()
        batch_data_left_mask = torch.stack([torch.tensor(x.input_mask) for x in batch_data_left], dim=0).long()
        batch_data_left_pos = torch.stack([torch.tensor(x.pos) for x in batch_data_left], dim=0).long()

        batch_data_right_word = torch.stack([torch.tensor(x.input_ids) for x in batch_data_right], dim=0).long()
        batch_data_right_mask = torch.stack([torch.tensor(x.input_mask) for x in batch_data_right], dim=0).long()
        batch_data_right_pos = torch.stack([torch.tensor(x.pos) for x in batch_data_right], dim=0).long()

        return (batch_data_left_word, batch_data_left_mask, batch_data_left_pos), \
            (batch_data_right_word, batch_data_right_mask, batch_data_right_pos)

    def _data_(self):
        all_word = torch.stack([torch.tensor(x.input_ids) for x in self.data], dim=0).long()
        all_mask = torch.stack([torch.tensor(x.input_mask) for x in self.data], dim=0).long()
        all_pos = torch.stack([torch.tensor(x.pos) for x in self.data], dim=0).long()
        labels = torch.stack([torch.tensor(x.label_id) for x in self.data], dim=0).long()
        data_to_cluster = TensorDataset(all_word, all_mask, all_pos)
        return data_to_cluster, labels