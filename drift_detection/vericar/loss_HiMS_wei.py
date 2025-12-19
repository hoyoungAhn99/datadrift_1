import torch
import torch.nn.functional as F


def get_slice_distance_weights(slice_labels, scale=2.0, dist_pow=1.0):

    B, current_depth = slice_labels.size()

    matches = (slice_labels.unsqueeze(1) == slice_labels.unsqueeze(0)).float()

    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)  # [B, B]

    tree_dist = float(current_depth) - shared_depth
    tree_dist = tree_dist**dist_pow

    weights = torch.pow(scale, tree_dist - 1)

    return weights


def ms_loss_level(sim_mat, labels, neg_weights, alpha=2.0, beta=50.0, lam=0.5):

    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)

    loss_list = []

    for i in range(batch_size):
        label_i = labels[i]

        # 현재 슬라이스 기준 Positive / Negative
        mask_pos = labels == label_i
        mask_neg = labels != label_i

        mask_pos[i] = False  # 자기 자신 제외

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]

        current_neg_w = neg_weights[i][mask_neg]

        if len(pos_sim) > 0:
            pos_term = (
                torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
            )
        else:
            pos_term = torch.tensor(0.0, device=device)

        if len(neg_sim) > 0:
            weighted_exp_sum = torch.sum(
                current_neg_w * torch.exp(beta * (neg_sim - lam))
            )
            neg_term = torch.log(1.0 + weighted_exp_sum) / beta
        else:
            neg_term = torch.tensor(0.0, device=device)

        loss_list.append(pos_term + neg_term)

    return torch.stack(loss_list)  # [B]


def HiMS_min_wei_loss(
    features,
    hi_labels,
    batch_size,
    num_hi,
    alpha=2.0,
    beta=50.0,
    lam=0.5,
    dist_scale=2.0,
    dist_pow=1.0,
):
    sim_mat = torch.matmul(features, features.t())  # [B, B]

    B = batch_size
    L = num_hi
    device = features.device

    level_losses = []

    for level in range(L):
        current_hierarchy_path = hi_labels[:, 0 : level + 1]

        _, labels_level = torch.unique(
            current_hierarchy_path, dim=0, return_inverse=True
        )

        current_weights = get_slice_distance_weights(
            current_hierarchy_path, scale=dist_scale, dist_pow=dist_pow
        )

        loss_l = ms_loss_level(
            sim_mat,
            labels_level,
            current_weights,
            alpha=alpha,
            beta=beta,
            lam=lam,
        )
        level_losses.append(loss_l)

    total_loss = 0.0
    L_float = float(L)

    cur_min = None
    for rev_level in range(L - 1, -1, -1):
        l = rev_level
        base_loss_l = level_losses[l]

        if cur_min is None:
            cur_loss = base_loss_l
        else:
            cur_loss = torch.min(base_loss_l, cur_min)

        cur_min = cur_loss

        k_l = torch.exp(torch.tensor(1.0 / (l + 1.0), device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / L_float

    return total_loss
