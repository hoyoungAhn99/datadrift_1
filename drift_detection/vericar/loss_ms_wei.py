import torch
import torch.nn.functional as F


def get_tree_distance_matrix(hi_labels, num_hi):
    B, L = hi_labels.size()

    matches = (hi_labels.unsqueeze(1) == hi_labels.unsqueeze(0)).float()

    continuous_matches = torch.cumprod(matches, dim=2)

    shared_depth = continuous_matches.sum(dim=2)

    tree_dist = float(L) - shared_depth

    return tree_dist


def weighted_ms_loss(sim_mat, tree_dist, alpha=2.0, beta=50.0, lam=0.5, dist_scale=1.5):
    device = sim_mat.device
    batch_size = sim_mat.size(0)

    loss_list = []

    diag_mask = torch.eye(batch_size, device=device).bool()

    for i in range(batch_size):
        mask_pos = (tree_dist[i] == 0) & (~diag_mask[i])
        pos_sim = sim_mat[i][mask_pos]

        mask_neg = tree_dist[i] > 0
        neg_sim = sim_mat[i][mask_neg]
        neg_dist = tree_dist[i][mask_neg]

        neg_weights = torch.pow(dist_scale, neg_dist)


        if len(pos_sim) > 0:
            pos_term = (
                torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))) / alpha
            )
        else:
            pos_term = torch.tensor(0.0, device=device)

        if len(neg_sim) > 0:
            neg_term_weighted = torch.sum(
                neg_weights * torch.exp(beta * (neg_sim - lam))
            )

            neg_term = torch.log(1.0 + neg_term_weighted) / beta
        else:
            neg_term = torch.tensor(0.0, device=device)

        loss_list.append(pos_term + neg_term)

    return torch.stack(loss_list).mean()


def ms_wei_loss(
    features,
    hi_labels,
    batch_size,
    num_hi,
    alpha=2.0,
    beta=50.0,
    lam=0.5,
    dist_scale=2.0,
):
    sim_mat = torch.matmul(features, features.t())

    tree_dist_mat = get_tree_distance_matrix(hi_labels, num_hi)

    total_loss = weighted_ms_loss(
        sim_mat,
        tree_dist_mat,
        alpha=alpha,
        beta=beta,
        lam=lam,
        dist_scale=dist_scale,
    )

    return total_loss
