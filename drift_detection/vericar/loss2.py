import torch
import torch.nn.functional as F

def ms_loss_level(sim_mat, labels, alpha=2.0, beta=50.0, lam=0.5):
    # 이 함수는 수정할 필요가 없습니다. 
    # 입력받는 labels가 이미 상위 계층 정보를 포함하도록 바깥에서 처리해서 넘겨줄 것입니다.
    device = sim_mat.device
    batch_size = sim_mat.size(0)
    labels = labels.view(-1)

    loss_list = []

    for i in range(batch_size):
        label_i = labels[i]
        # labels가 상위 계층까지 고려한 고유 ID이므로, 
        # 여기서 (labels == label_i)는 "루트부터 현재 레벨까지 모두 같은 샘플"만 True가 됩니다.
        mask_pos = (labels == label_i) 
        mask_neg = (labels != label_i)
        mask_pos[i] = False

        pos_sim = sim_mat[i][mask_pos]
        neg_sim = sim_mat[i][mask_neg]
        
        # Positive Pair가 하나도 없는 경우(자기 자신 제외)에 대한 예외 처리가 있으면 더 안전합니다.
        # 여기서는 원본 로직을 유지합니다.
        
        # Pos term 계산
        if len(pos_sim) > 0:
            pos_term = torch.log(
                1.0 + torch.sum(torch.exp(-alpha * (pos_sim - lam)))
            ) / alpha
        else:
            pos_term = torch.tensor(0.0, device=device)

        # Neg term 계산
        if len(neg_sim) > 0:
            neg_term = torch.log(
                1.0 + torch.sum(torch.exp(beta * (neg_sim - lam)))
            ) / beta
        else:
            neg_term = torch.tensor(0.0, device=device)

        loss_list.append(pos_term + neg_term)

    return torch.stack(loss_list) # [B]


def HiMS_min_loss(features, hi_labels, batch_size, num_hi,
                  alpha=2.0, beta=50.0, lam=0.5):
    """
    features : [B, D]
    hi_labels: [B, L] 
               (L = num_hi)
               Assumes columns are ordered from Root -> Leaf
               ex) index 0: Type (Highest), ..., index L-1: Year (Lowest)
    """
    sim_mat = torch.matmul(features, features.t())  # [B, B]

    B = batch_size
    L = num_hi
    device = features.device

    level_losses = []
    
    # [수정된 부분]: Loop logic changed to incorporate hierarchy
    for level in range(L):
        # 기존: labels_level = hi_labels[:, level] -> 해당 컬럼만 봄
        
        # 수정: Root(0)부터 현재 level까지 slicing ([B, level+1])
        # 예: level=3이면 0,1,2,3 컬럼을 모두 가져옴 (Type, Make, Model, Year)
        current_hierarchy_path = hi_labels[:, 0 : level + 1]
        
        # torch.unique를 사용하여 (Type, Make, Model, Year) 조합이 같은 행들에 
        # 동일한 정수 ID(inverse_indices)를 부여합니다.
        # 이렇게 하면 ms_loss_level 입장에서는 단순한 1D 라벨처럼 보이지만,
        # 실제로는 상위 계층 조건이 모두 포함된 라벨이 됩니다.
        _, labels_level = torch.unique(current_hierarchy_path, dim=0, return_inverse=True)
        
        loss_l = ms_loss_level(sim_mat, labels_level,
                               alpha=alpha, beta=beta, lam=lam)
        level_losses.append(loss_l)

    # 아래 로직(Min-pooling & Weighting)은 그대로 유지
    total_loss = 0.0
    L_float = float(L)

    cur_min = None
    for rev_level in range(L - 1, -1, -1):
        l = rev_level
        base_loss_l = level_losses[l] # Current level's per-sample loss, shape [B]

        if cur_min is None:
            cur_loss = base_loss_l
        else:
            # 하위 레벨 loss가 상위 레벨 loss보다 커지지 않도록 clamp (논문의 HiConE 개념과 유사)
            cur_loss = torch.min(base_loss_l, cur_min)
        
        cur_min = torch.min(cur_loss)

        # 계층별 가중치 적용 (k_l)
        k_l = torch.exp(torch.tensor(1.0 / (l + 1.0), device=device))
        total_loss = total_loss + (k_l * cur_loss.mean()) / L_float

    return total_loss