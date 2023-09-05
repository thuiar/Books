from tools.utils import *

class SentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, pool_type='avg'):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.pool_type = pool_type
        self.out_dim = 768
        self.dense = nn.Linear(3*self.out_dim, self.out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, w, mask, pos, out_ht = False):
        x = self.bert(w, attention_mask=mask)[0] # b l h
        return self.get_pos_state(x, pos, out_ht)
    def get_pos_state(self,x, pos, out_ht= False):
        h = torch.zeros(x.size(0), x.size(1)).to(x.device)
        t = torch.zeros(x.size(0), x.size(1)).to(x.device)
        for i in range(x.size(0)):
            h[i, pos[i][0]:pos[i][1]] = 1.0
            t[i, pos[i][2]:pos[i][3]] = 1.0
        
        out = self.get_state_from_ht(x, h, t)
        if out_ht:
            return out, h,t, x
        return out
    def get_entites(self, w, mask, pos):
        x = self.bert(w, attention_mask=mask)[0] # b l h
        h = torch.zeros(x.size(0), x.size(1)).to(x.device)
        t = torch.zeros(x.size(0), x.size(1)).to(x.device)
        for i in range(x.size(0)):
            h[i, pos[i][0]:pos[i][1]] = 1.0
            t[i, pos[i][2]:pos[i][3]] = 1.0
        e1 = self.entity_trans(x, h)
        e2 = self.entity_trans(x, t)
        out = torch.cat([e1, e2], -1)
        return out
    def get_state_from_ht(self, x, h, t):
        e1 = self.entity_trans(x, h)
        e2 = self.entity_trans(x, t)
        out = self.dense(torch.cat([x[:, 0], e1, e2], -1))
        out = self.activation(out)
        out = self.dropout(out)
        return out
    def entity_trans(self, x, pos):
        e1 = x * pos.unsqueeze(2).expand(-1, -1, x.size(2))
        if self.pool_type == 'avg':
            divied = torch.sum(pos, 1)
            e1 = torch.sum(e1, 1) / divied.unsqueeze(1)
        return e1
    def forward_with_per(self, w, mask, pos, per=None):
        embed = self.bert.embeddings.word_embeddings(w)
        if per is not None:
            embed_ = embed + per
        else:
            embed_ = embed
        x = self.bert(inputs_embeds = embed_, attention_mask=mask)[0]
        state = self.get_pos_state(x, pos)
        return embed, state#, x

# class SentenceEncoder(nn.Module):
#     def __init__(self, pretrain_path, activation = 'relu'):
#         nn.Module.__init__(self)
#         self.bert = BertModel.from_pretrained(pretrain_path)
#         self.hidden_size = 768
#         if activation == 'relu':
#             self.fc = nn.Sequential(
#                 nn.Linear(2 * self.hidden_size, self.hidden_size),
#                 nn.ReLU()
#             )
#         else:
#             self.fc = nn.Sequential(
#                 nn.Linear(2 * self.hidden_size, self.hidden_size),
#                 nn.Tanh()
#             )
#         self.out_dim = 768

#     def forward(self, w, mask, pos, need_ori = False):
#         x = self.bert(w, attention_mask=mask)[0]
#         state = self._get_pos_state(x, pos)
#         if need_ori:
#             return state, x
#         return state#, x
#     def forward_with_per(self, w, mask, pos, per=None):
#         embed = self.bert.embeddings.word_embeddings(w)
#         if per is not None:
#             embed_ = embed + per
#         else:
#             embed_ = embed
#         x = self.bert(inputs_embeds = embed_, attention_mask=mask)[0]
#         state = self._get_pos_state(x, pos)
#         return embed, state#, x
#     def _get_pos_state(self, x, pos):
#         tensor_range = torch.arange(x.size()[0])
#         e1s_state = x[tensor_range, pos[:, 0]]
#         # e1e_state = x[tensor_range, pos[:, 1]]
#         e2s_state = x[tensor_range, pos[:, 2]]
#         # e2e_state = x[tensor_range, pos[:, 3]]
#         state = torch.cat((e1s_state, e2s_state), -1)
#         state = self.fc(state)
#         return state

class SentenceEncoderWithHT(nn.Module):
    def __init__(self, pretrain_path):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.out_dim = 768*2

    def forward(self, w, mask, pos):
        x = self.bert(w, attention_mask=mask)[0]
        # pos = inputs['pos']
        tensor_range = torch.arange(w.size()[0])

        e1s_state = x[tensor_range, pos[:, 0]]

        e2s_state = x[tensor_range, pos[:, 2]]

        state = torch.cat((e1s_state, e2s_state), -1)
        return state#, x