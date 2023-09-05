from relation_discover.utils import *

class SentenceEncoderWithHT(nn.Module):
    def __init__(self, pretrain_path):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.out_dim = 768*2

    def forward(self, w, mask, pos):
        x = self.bert(w, attention_mask=mask)[0]

        tensor_range = torch.arange(w.size()[0])

        e1s_state = x[tensor_range, pos[:, 0]]

        e2s_state = x[tensor_range, pos[:, 2]]

        state = torch.cat((e1s_state, e2s_state), -1)
        return state


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        pretrain_path=args.bert_path
        label_num=args.rel_nums
        self.rel_nums=args.n_clusters
        z_dim=args.z_dim
        self.pos_margin = args.pos_margin # 0.7
        self.neg_margin = args.neg_margin # 1.4
        self.temp = args.temp
        self.encoder = SentenceEncoderWithHT(pretrain_path)
        hidden_size = self.encoder.out_dim
        self.fc = nn.Linear(hidden_size, args.z_dim)

    def forward(self, batch_input, labels=None, unlabel=False):

        pos = batch_input['pos']
        mask = batch_input['mask']
        w = batch_input['word']
        x = self.encoder(w, mask, pos)
        h = self.fc(x)
        h = F.normalize(h, p=2, dim=1)
        if labels is not None:
            loss = self.loss(h, labels, unlabel=unlabel)
            return loss
        return h

    def get_h(self, batch_input):
        pos = batch_input['pos']
        mask = batch_input['mask']
        w = batch_input['word']

        x = self.encoder(w, mask, pos)
        h = self.fc(x)
        h = F.normalize(h, p=2, dim=1)
        return h
    def loss(self, x, labels, unlabel=False):
        dist_mat = _dist_(x, x)
        pos_margin =self.pos_margin
        neg_margin =self.neg_margin
        a = self.temp
        b = self.temp
        N = dist_mat.size(0)
        labels = labels[0:N]
        total_loss = Variable(torch.tensor(0.0), requires_grad = True)

        for ind in range(N):
            
            is_pos = labels.eq(labels[ind])
            is_pos[ind] = 0
            is_neg = labels.ne(labels[ind])

            dist_ap = dist_mat[ind][0:N][is_pos]
            dist_an = dist_mat[ind][0:N][is_neg]

            # pos
            gama_p = torch.lt(-dist_ap, -pos_margin)
            gama_lt = dist_ap[gama_p]
            gama = (gama_lt - pos_margin)**2

            # neg
            beta_n = torch.lt(dist_an, neg_margin)
            beta_lt = dist_an[beta_n]
            beta = (neg_margin -beta_lt)**2
            if len(gama)>0:
                gama = torch.logsumexp(a * gama, dim=0)
            else:
                gama = 0.0
            if len(beta)>0:
                beta = torch.logsumexp(b * beta, dim=0)
            else:
                beta = 0.0
            loss = gama + beta
            total_loss = total_loss + loss
        total_loss = total_loss / N

        return total_loss

def _dist_(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())

    error_mask = pairwise_distances_squared <= 0.0

    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances  