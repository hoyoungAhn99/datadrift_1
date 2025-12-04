import torch
import torch.nn.functional as F

def MS_loss(features, hi_labels, alpha=2.0, beta=50.0, lam=0.5):
    """
    features : [B, D]
    hi_labels: [B, L]
    """
    batch_size = features.size(0)
    
    # 2. Compute Similarity Matrix
    sim_mat = torch.matmul(features, features.t()) 
    
    # Clamp for numerical stability (avoids slightly > 1.0 or < -1.0 due to float errors)
    sim_mat = torch.clamp(sim_mat, min=-1.0 + 1e-6, max=1.0 - 1e-6)

    # 3. Define Positives (Same Make-Type-Model-Year class) 
    # Check if all hierarchy levels match
    label_comparison = (hi_labels.unsqueeze(1) == hi_labels.unsqueeze(0))
    mask_pos = label_comparison.all(dim=2) 
    
    # 4. Define Negatives (Different Make-Type-Model-Year class) 
    # Anything that is NOT a positive is a negative.
    mask_neg = ~mask_pos
    
    # 5. Mask out self-similarity (Diagonal)
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    mask_pos[diag_mask] = False
    mask_neg[diag_mask] = False

    loss = []
    
    for i in range(batch_size):
        pos_sim_i = sim_mat[i][mask_pos[i]]
        neg_sim_i = sim_mat[i][mask_neg[i]]

        # Standard MS Loss Calculation (Eq. 1) 
        # 1/alpha * log(1 + sum(exp(-alpha * (S_pos - lambda))))
        if pos_sim_i.numel() > 0:
            pos_term = (1.0 / alpha) * torch.log(
                1.0 + torch.sum(torch.exp(-alpha * (pos_sim_i - lam)))
            )
        else:
            pos_term = torch.tensor(0.0, device=features.device)

        # 1/beta * log(1 + sum(exp(beta * (S_neg - lambda))))
        if neg_sim_i.numel() > 0:
            neg_term = (1.0 / beta) * torch.log(
                1.0 + torch.sum(torch.exp(beta * (neg_sim_i - lam)))
            )
        else:
            neg_term = torch.tensor(0.0, device=features.device)
        
        loss.append(pos_term + neg_term)

    return torch.stack(loss).mean()