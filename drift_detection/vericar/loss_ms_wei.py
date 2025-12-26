import torch
import torch.nn.functional as F

def get_tree_distance_matrix(hi_labels, num_hi):
    B, L = hi_labels.size()
    matches = (hi_labels.unsqueeze(1) == hi_labels.unsqueeze(0)).float()
    continuous_matches = torch.cumprod(matches, dim=2)
    shared_depth = continuous_matches.sum(dim=2)
    tree_dist = float(L) - shared_depth
    return tree_dist

def weighted_ms_loss_mined(sim_mat, tree_dist, alpha=2.0, beta=50.0, lam=0.5, dist_scale=1.5, mining_margin=0.1):
    """
    Hard Mining이 추가된 Weighted Multi-Similarity Loss
    """
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    loss_list = []
    
    # 자기 자신(Diagonal) 마스킹을 위한 준비
    diag_mask = torch.eye(batch_size, device=device).bool()

    for i in range(batch_size):
        # 1. 전체 Positive / Negative 후보군 추출
        # Positive: Tree Distance가 0인 경우 (완전 일치)
        mask_pos = (tree_dist[i] == 0) & (~diag_mask[i])
        pos_sim = sim_mat[i][mask_pos]

        # Negative: Tree Distance가 0보다 큰 경우 (다른 클래스)
        mask_neg = tree_dist[i] > 0
        neg_sim = sim_mat[i][mask_neg]
        neg_dist = tree_dist[i][mask_neg] # Negative들의 Tree Distance도 같이 가져옴

        # 예외 처리: Positive나 Negative가 아예 없는 경우 스킵
        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue

        # 2. Hardest Pair 탐색
        # 가장 안 닮은 Positive (유사도 최소값)
        hardest_pos_sim = torch.min(pos_sim)
        # 가장 닮은 Negative (유사도 최대값)
        hardest_neg_sim = torch.max(neg_sim)

        # 3. Hard Mining (Filtering)
        # Negative Mining: 가장 먼 Positive보다 더 가깝거나 헷갈리는 Negative만 선택
        # 조건: neg_sim + margin > hardest_pos_sim
        mining_idx_neg = (neg_sim + mining_margin) > hardest_pos_sim
        neg_sim_mined = neg_sim[mining_idx_neg]
        neg_dist_mined = neg_dist[mining_idx_neg] # Distance도 같이 필터링

        # Positive Mining: 가장 가까운 Negative보다 더 멀거나 헷갈리는 Positive만 선택
        # 조건: pos_sim - margin < hardest_neg_sim
        mining_idx_pos = (pos_sim - mining_margin) < hardest_neg_sim
        pos_sim_mined = pos_sim[mining_idx_pos]

        # Mining 후 남은 샘플이 없으면 해당 앵커는 이미 학습이 잘 된 것이므로 Loss 0 처리 (Skip)
        if pos_sim_mined.numel() == 0 or neg_sim_mined.numel() == 0:
            continue

        # 4. Loss Term 계산 (Mined Samples 사용)
        
        # Positive Term (Standard MS Loss formula)
        pos_term = (
            torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim_mined - lam)))) / alpha
        )

        # Negative Term (Weighted MS Loss formula)
        # 필터링된 Negative들에 대해서만 Weight 계산
        neg_weights = torch.pow(dist_scale, neg_dist_mined)
        
        neg_term_weighted = torch.sum(
            neg_weights * torch.exp(beta * (neg_sim_mined - lam))
        )
        neg_term = torch.log(1.0 + neg_term_weighted) / beta

        loss_list.append(pos_term + neg_term)

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

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
    mining_margin=0.1, # Mining Margin 파라미터 추가
):
    # Cosine Similarity Matrix 계산
    sim_mat = torch.matmul(features, features.t())
    # 수치적 안정성을 위해 clamp
    sim_mat = torch.clamp(sim_mat, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    tree_dist_mat = get_tree_distance_matrix(hi_labels, num_hi)

    total_loss = weighted_ms_loss_mined(
        sim_mat,
        tree_dist_mat,
        alpha=alpha,
        beta=beta,
        lam=lam,
        dist_scale=dist_scale,
        mining_margin=mining_margin
    )

    return total_loss