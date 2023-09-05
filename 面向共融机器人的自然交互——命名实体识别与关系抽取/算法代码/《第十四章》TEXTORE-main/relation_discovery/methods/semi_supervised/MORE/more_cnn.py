from tools.utils import *
from .loss import metric_loss, pairwise_distance

class Embedding_word(nn.Module):

    def __init__(self, vocab_size, embedding_dim, weights, requires_grad=True):
        """
        the weights will be add one random vector for unk word
                add one zeros vector for blank word
                all weights should be trainable.
        :param vocab_size:
        :param embedding_dim:
        :param weights: the numpy version.
        :param trainable:
        """
        super(Embedding_word, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.word_embedding.weight.data.copy_(torch.from_numpy(weights))
        self.word_embedding.weight.requires_grad = requires_grad

    def forward(self, idx_input):
        return self.word_embedding(idx_input)
class CNN(nn.Module):
    def __init__(self, cnn_input_shape, out_dim=64):
        """
        :param cnn_input_shape: [max_len, word_embedding+2 * pos_emb]
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=cnn_input_shape[-1], out_channels=230, kernel_size=3)
        self.max_pool1 = nn.MaxPool1d(kernel_size=cnn_input_shape[0] - 2)

        self.linear = nn.Linear(230, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = x.reshape(-1, 230)

        return self.linear(x)
class PCnn(nn.Module):
    def __init__(self, opt):
        super(PCnn, self).__init__()
        word_vec_mat = opt.glove_mat
        max_len = opt.max_length
        pos_emb_dim = opt.pos_emb_dim
        dropout = opt.drop_out
        out_dim = opt.embedding_dim
        # defining layers
        dict_shape = word_vec_mat.shape
        self.word_emb = Embedding_word(dict_shape[0], dict_shape[1], weights=word_vec_mat,
                                                  requires_grad=True)
        # default trainable.
        self.pos1_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.pos2_emb = nn.Embedding(max_len * 2, pos_emb_dim)
        self.drop = nn.Dropout(p=dropout)

        cnn_input_shape = (max_len, dict_shape[1] + 2 * pos_emb_dim)
        self.convnet = CNN(cnn_input_shape, out_dim)

        self.ml = metric_loss()

    def forward(self, batch_input, perturbation=None):
        pos1 = batch_input['pos1']
        pos2 = batch_input['pos2']
        word = batch_input['word']

        pos1_emb = self.pos1_emb(pos1)
        pos2_emb = self.pos2_emb(pos2)
        word_emb = self.word_emb(word)

        drop = self.drop(word_emb)
        if perturbation is not None:
            drop += perturbation

        cnn_input = torch.cat([drop, pos1_emb, pos2_emb], -1)
        cnn_input = cnn_input.permute([0, 2, 1])  # [B, embedding, max_len]
        encoded = self.convnet(cnn_input)

        return word_emb, encoded

    def forward_norm(self, batch_input, pertubation=None):
        word_emb, encoded = self.forward(batch_input, pertubation)
        encoded = self.norm(encoded)
        return word_emb, encoded

    def norm(self, encoded):
        return F.normalize(encoded, p=2, dim=1)

    def set_embedding_weight(self, weight):
        self.word_emb.word_embedding.weight.data[:-2].copy_(torch.from_numpy(weight))
    def get_params(self, lr):
        self.lr = lr
        params = []
        params += [{'params': self.word_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos1_emb.parameters(), 'lr': self.lr}]
        params += [{'params': self.pos2_emb.parameters(), 'lr': self.lr}]
        params += [{'params': list(self.convnet.conv1.parameters())[0], 'weight_decay': 2e-4, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.conv1.parameters())[0]}]
        params += [{'params': list(self.convnet.conv1.parameters())[1], 'lr': self.lr}]

        params += [{'params': list(self.convnet.linear.parameters())[0], 'weight_decay': 1e-3, 'lr': self.lr}]
        # params += [{'params': list(self.convnet.linear.parameters())[0]}]
        params += [{'params': list(self.convnet.linear.parameters())[1], 'lr': self.lr}]
        return params