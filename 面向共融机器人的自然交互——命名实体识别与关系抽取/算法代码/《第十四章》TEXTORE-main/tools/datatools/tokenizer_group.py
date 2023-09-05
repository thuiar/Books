from tools.utils import *
BERT_MAX_LEN = 512


class CNNTokenizerForRE(object):
    unk = '[UNK]'
    pad = '[PAD]'
    head = 'HEAD'
    tail = 'TAIL'
    def __init__(self, word2id, max_length = 120):
        self.max_length = max_length
        self.word2id = word2id
    def __getraw__(self, item):
        pos1 = [item['head']['e1_begin'], item['head']['e1_end']]
        pos2 = [item['tail']['e2_begin'], item['tail']['e2_end']]
        word, pos1, pos2, pos, mask = self.tokenize(item['sentence'], pos1, pos2)
        return word, pos1, pos2, pos, mask

    def __additem__(self, d, word, pos1, pos2, pos = None, mask=None, mask_entity = False):
        
        if mask_entity and pos is not None:
            word = word.copy()
            word[pos[0]:pos[1]+1] = [self.word2id[self.head]]*(pos[1] - pos[0] + 1)
            word[pos[2]:pos[3]+1] = [self.word2id[self.tail]]*(pos[3] - pos[2] + 1)
            d['word'].append(word)
            d['pos1'].append(pos1)
            d['pos2'].append(pos2)
            d['pos'].append(pos)
            d['mask'].append(mask)
        else:
            d['word'].append(word)
            d['pos1'].append(pos1)
            d['pos2'].append(pos2)
            d['pos'].append(pos)
            d['mask'].append(mask)
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id[self.unk])
        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id[self.pad])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length
        pos = [pos_head[0], pos_head[-1], pos_tail[0], pos_tail[-1]]
        # mask
        # mask = np.zeros((self.max_length), dtype=np.int32)
        # mask[:len(indexed_tokens)] = 1
        # , pos, mask
        return indexed_tokens, pos1, pos2

class BertTokenizerForRE(object):
    def __init__(self, pretrain_path, max_length = 120, sf=False):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.sf = sf
    def __getraw__(self, item):
        pos1 = [item['head']['e1_begin'], item['head']['e1_end']]
        pos2 = [item['tail']['e2_begin'], item['tail']['e2_end']]
        word, mask, pos = self.tokenize(item['sentence'], pos1, pos2)
        return word, mask, pos
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        tokens = []
        cur_pos = 0

        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1_start_index = len(tokens)
                # tokens.append('[HEADSTART]')
            if cur_pos == pos_tail[0]:
                pos2_start_index = len(tokens)
                # tokens.append('[TAILSTART]')
            
            tokens += self.tokenizer.tokenize(token)

            if not self.sf:
                if cur_pos == pos_head[-1]:
                    pos1_end_index = len(tokens)
                    # tokens.append('[HEADEND]')
                    
                if cur_pos == pos_tail[-1]:
                    pos2_end_index = len(tokens)
                    # tokens.append('[TAILEND]')
            else:
                if cur_pos == pos_head[-1]-1:
                    pos1_end_index = len(tokens)
                    # tokens.append('[HEADEND]')
                    
                if cur_pos == pos_tail[-1]-1:
                    pos2_end_index = len(tokens)
                    # tokens.append('[TAILEND]')
            cur_pos += 1
        tokens.append('[SEP]')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        pos = [pos1_start_index, pos1_end_index, pos2_start_index, pos2_end_index]
        pos = [x if x < self.max_length else self.max_length-1 for x in pos]

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask, pos
    def __additem__(self, d, word, pos = None, mask=None, mask_entity = False):
        
        if mask_entity and pos is not None:
            word = word.copy()
            word[pos[0]:pos[1]+1] = [self.word2id[self.head]]*(pos[1] - pos[0] + 1)
            word[pos[2]:pos[3]+1] = [self.word2id[self.tail]]*(pos[3] - pos[2] + 1)
            d['word'].append(word)
            d['pos'].append(pos)
            d['mask'].append(mask)
        else:
            d['word'].append(word)
            d['pos'].append(pos)
            d['mask'].append(mask)
    def get_len(self, item):
        # pos_head = [item['head']['e1_begin'], item['head']['e1_end']]
        # pos_tail = [item['tail']['e2_begin'], item['tail']['e2_end']]
        raw_tokens, pos_head, pos_tail = item['tokens'], item['h'][2][0], item['t'][2][0]

        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0

        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[HEADSTART]')
                pos1_start_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[TAILSTART]')
                pos2_start_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                pos1_end_index = len(tokens)
                tokens.append('[HEADEND]')
                
            if cur_pos == pos_tail[-1]:
                pos2_end_index = len(tokens)
                tokens.append('[TAILEND]')
                
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return len(indexed_tokens)

def get_tokenizer(args):
    if args.backbone == 'cnn':
        try:
            args.glove_mat = np.load(os.path.join(args.glove_path, 'glove_mat.npy'))
            glove_word2id = json.load(open(os.path.join(args.glove_path,'glove_word2id.json')))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        tokenizer = CNNTokenizerForRE(glove_word2id, args.max_length)
    elif args.backbone == 'bert':

        tokenizer = BertTokenizerForRE(
            args.bert_path,
            args.max_length,
            sf = args.sf
        )
    else:
        raise NotImplementedError
    return tokenizer