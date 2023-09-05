from tools.utils import *


pos_margin =  0.7
neg_margin = 1.4
temperature = 50

def pos_inf(dtype):
    return torch.finfo(dtype).max


def neg_inf(dtype):
    return torch.finfo(dtype).min

# input must be 2D
def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        mk = torch.any(keep_mask, dim=dim, keepdim=True)
        output = output.masked_fill(~mk, 0)
        div = mk.long().sum()
        if div>0:
            output = output.sum() / div
        else:
            output = Variable(torch.tensor(0.0), requires_grad = True)
    return output
def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())

    error_mask = pairwise_distances_squared <= 0.0

    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances   

def pml_loss(x, lb):
    # x bxb
    pos_margin =  0.7
    neg_margin = 1.4
    temperature = 50
    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
    pos_mask = (lb.unsqueeze(1).repeat(1, lb.shape[0]) == lb).to(device).long() # n*m
    neg_mask = 1 - pos_mask
    pos_mask = pos_mask - torch.eye(pos_mask.size(0)).to(device)
    pos_info = (x > pos_margin).long()
    neg_info = (x < neg_margin).long()
    gama = (x - pos_margin)**2
    beta = (neg_margin - x)**2
    gama_loss = logsumexp(gama * temperature, keep_mask=(pos_mask * pos_info).bool(), add_one=False, dim=1)
    beta_loss = logsumexp(beta * temperature, keep_mask=(neg_mask * neg_info).bool(), add_one=False, dim=1)
    loss = gama_loss + beta_loss
    return loss

def pml_loss_with_proxy(x, aug_x, lb, num_class = 100):
    # x: bxd aug_x: bxcxd
    x_ = x.unsqueeze(1).expand(-1, aug_x.size(1), -1)
    dist = ((x_-aug_x)**2).sum(2).sqrt() # b*c
    pos_mask = F.one_hot(lb, num_classes=num_class) # b*c
    neg_mask = 1 - pos_mask
    pos_info = (dist > pos_margin).long()
    neg_info = (dist < neg_margin).long()
    gama = (dist - pos_margin)**2
    beta = (neg_margin - dist)**2
    gama_loss = logsumexp(gama * temperature, keep_mask=(pos_mask * pos_info).bool(), add_one=False, dim=1)
    beta_loss = logsumexp(beta * temperature, keep_mask=(neg_mask * neg_info).bool(), add_one=False, dim=1)
    loss = gama_loss + beta_loss
    return loss

def pml_loss_with_pair(x, aug_x, lb):
    # x: bxd aug_x: bxbxd
    x_ = x.unsqueeze(1).expand(-1, aug_x.size(1), -1)
    dist = ((x_-aug_x)**2).sum(2).sqrt() # b*c
    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
    pos_mask = (lb.unsqueeze(1).repeat(1, lb.shape[0]) == lb).to(device).long() # n*m
    neg_mask = 1 - pos_mask
    pos_mask = pos_mask - torch.eye(pos_mask.size(0)).to(device)
    pos_info = (dist > pos_margin).long()
    neg_info = (dist < neg_margin).long()
    gama = (dist - pos_margin)**2
    beta = (neg_margin - dist)**2
    gama_loss = logsumexp(gama * temperature, keep_mask=(pos_mask * pos_info).bool(), add_one=False, dim=1)
    beta_loss = logsumexp(beta * temperature, keep_mask=(neg_mask * neg_info).bool(), add_one=False, dim=1)
    loss = gama_loss + beta_loss
    return loss

def adapate_loss_with_proxy(x, aug_x, lb, num_labels):
    # x: bxd aug_x: bxbxd
    th = torch.norm(x, p=2, dim=1, keepdim=True)
    aug_x = torch.norm(aug_x, p=2, dim=2)
    logits = torch.cat([aug_x, th], dim=1)

    labels = F.one_hot(lb, num_classes=num_labels+1)
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, -1] = 1.0
    labels[:, -1] = 0.0

    p_mask = labels + th_label
    n_mask = 1 - labels
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

    loss = loss1 + loss2
    loss = loss.mean()
    return loss