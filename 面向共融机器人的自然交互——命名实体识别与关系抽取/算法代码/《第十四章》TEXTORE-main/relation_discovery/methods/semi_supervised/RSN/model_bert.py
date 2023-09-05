from tools.utils import *
from .loss import sigmoid_cross_entropy_loss, sigmoid_cond_loss

class BERTModel(nn.Module):
    def __init__(self, args, device):
        super(BERTModel, self).__init__()
        self.device = device
        max_len = args.max_length
        out_dim = args.out_dim
        self.p_mult = args.p_mult
        # defining layers
        self.encoder = SentenceEncoder(args.bert_path)
        self.p = nn.Linear(self.encoder.out_dim, 1)
    
    def forward(self, word_left, pos1_left, pos2_left,
                word_right, pos1_right, pos2_right, perturbation1=None, perturbation2=None):
        left_word_emb, encoded_l = self.forward_bert(word_left, pos1_left, pos2_left, perturbation1)
        right_word_emd, encoded_r = self.forward_bert(word_right, pos1_right, pos2_right, perturbation2)
        l1_dist = torch.abs(encoded_l - encoded_r)
        prediction = self.p(l1_dist)
        prediction = torch.sigmoid(prediction).squeeze(1)
        return prediction, left_word_emb, right_word_emd, encoded_l, encoded_r
    
    def get_hiddens(self, w, p1, p2):
        _, encoded = self.forward_bert(w, p1, p2)
        return encoded

    def forward_cnn(self, word, pos1, pos2, perturbation=None):

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
    
    def forward_bert(self, word, mask, pos, perturbation=None):

        word_emb, encoded = self.encoder.forward_with_per(word, mask, pos, perturbation)

        return word_emb, encoded

    def pred_X(self, h1, h2):
        h1 = torch.stack(h1, dim=0)
        h2 = torch.stack(h2, dim=0)
        l1_dist = torch.abs(h1 - h2)
        prediction = self.p(l1_dist)
        prediction = torch.sigmoid(prediction).squeeze(1)
        return prediction

    def validation(self, val_left_input, val_right_input, labels):
        pred, left_word_emb, right_word_emd, encoded_l, encoded_r = self.forward(*val_left_input, *val_right_input)
        hard_pred = (pred > 0.5).long()
        fp = torch.mean((hard_pred < labels).float())
        fn = torch.mean((hard_pred >= labels).float())
        tp = torch.mean(((labels==0) * (hard_pred==0)).float())
        tn = torch.mean(((labels==1)*(hard_pred==1)).float())
        tp=round(tp.item(),5)
        fp=round(fp.item(),5)
        fn=round(fn.item(),5)
        tn=round(tn.item(),5)
        acc = torch.mean((hard_pred == labels).float()).item()
        return acc, tp, fp, fn, tn
    
    def get_loss_and_emb(self, pred, labels):
        loss = sigmoid_cross_entropy_loss(pred,labels)
        return loss
    
    def get_cond_loss(self, pred):
        cond_loss = sigmoid_cond_loss(pred)
        return cond_loss
    
    def get_v_adv_loss(self, ul_left_input, ul_right_input, prob, shape, power_iterations=1):
        bernoulli = Bernoulli
        p_mult = self.p_mult
        prob = torch.clamp(prob, min=1e-7, max=1.-1e-7)
        prob_dist = bernoulli(probs=prob)
        #generate virtual adversarial perturbation

        left_d = torch.FloatTensor(shape).uniform_(0, 1).to(self.device)
        left_d.requires_grad = True
        right_d = torch.FloatTensor(shape).uniform_(0, 1).to(self.device)
        right_d.requires_grad = True
        for _ in range(power_iterations):
            # 扰动
            left_d = (0.02) * F.normalize(left_d, p=2, dim=1)
            right_d = (0.02) * F.normalize(right_d, p=2, dim=1)

            p_prob = self.forward(*ul_left_input, *ul_right_input, perturbation1= left_d,  perturbation2= right_d)[0]
            p_prob = torch.clamp(p_prob, min=1e-7, max=1.-1e-7)
            
            kl = kl_divergence(prob_dist, bernoulli(probs=p_prob)).mean()

            left_gradient, right_gradient = torch.autograd.grad(kl, [left_d,right_d], retain_graph=True)
            left_d, right_d = left_gradient, right_gradient

            left_d.requires_grad = False
            right_d.requires_grad = False

        left_d = p_mult * F.normalize(left_d, dim=1)
        right_d = p_mult * F.normalize(right_d, dim=1)
        #virtual adversarial loss
        prob = self.forward(*ul_left_input, *ul_right_input, left_d, right_d)[0]
        p_prob = torch.clamp(prob, 1e-7, 1.-1e-7)
        v_adv_losses = kl_divergence(prob_dist, bernoulli(probs=p_prob))
        return v_adv_losses.mean()
