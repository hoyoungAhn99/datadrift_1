import torch
import torch.nn.functional as F

def ms_loss_level(sim_mat, labels, alpha=2.0, beta=50.0, lam=0.5):
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)

    loss_list = []

    for i in range(batch_size):
        label_i = labels[i]
        mask_pos = (labels == label_i)
        mask_neg = (labels != label_i)
        mask_pos[i] = False

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]

        pos_term = torch.log(
            1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))
        ) / alpha

        neg_term = torch.log(
            1.0 + torch.sum(torch.exp(beta * (neg_sim - lam)))
        ) / beta

        loss_list.append(pos_term + neg_term)

    return torch.stack(loss_list) # [B]


def HiMS_min_loss(features, hi_labels, batch_size, num_hi,
                  alpha=2.0, beta=50.0, lam=0.5):
    """
    features : [B, D]
    hi_labels: [B, L] (L = num_hi, ex: year, model, make, type)
    """
    sim_mat = torch.matmul(features, features.t())  # [B, B]

    B = batch_size
    L = num_hi
    device = features.device

    level_losses = []
    for level in range(L):
        labels_level = hi_labels[:, level]
        loss_l = ms_loss_level(sim_mat, labels_level,
                               alpha=alpha, beta=beta, lam=lam)
        level_losses.append(loss_l)

    total_loss = 0.0
    L_float = float(L)

    cur_min = None
    for rev_level in range(L - 1, -1, -1):
        l = rev_level
        base_loss_l = level_losses[l] # Current level's per-sample loss, shape [B]

        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)
        
        cur_min = torch.min(cur_loss)

        k_l = torch.exp(torch.tensor(1.0 / (l + 1.0), device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / L_float

    return total_loss