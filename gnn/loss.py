import torch
import torch.nn.functional as F

def lossFunction(recon, target, mu, logvar, mask):
    """
    Role:
        - 마스킹된 노드에 대해서만 reconstruction loss와 KL divergence 계산

    Args:
        - recon, target: [num_ingre, ingre_out_dim]
        - mask: [num_ingre] boolean tensor, True인 위치는 마스킹된 노드
        - Reconstruction loss는 마스킹된 노드 feature에 대해 MSE로 계산.
        - KL divergence는 전체 잠재 벡터에 대해 계산.

    Tip:
        - recon_loss는 마스킹된 노드×출력차원 전부를 대상으로 평균이 됨
          (예: 노드 20개, 차원 10 → 200개의 원소 평균).

        - kl_loss 역시 잠재차원 전체를 대상으로 평균이 됨
          (예: 노드 20, latent_dim=16 → 320개의 원소 평균).
    """

    # 만약 마스킹된 노드가 없는 경우, recon[mask]는 빈 tensor가 될 수 있으므로,
    # 0 loss로 처리합니다.
    if mask.sum() > 0:
        #recon_loss = F.mse_loss(recon[mask], target[mask], reduction='sum')
        recon_loss = F.smooth_l1_loss(recon[mask], target[mask], reduction='sum')
    else:
        recon_loss = torch.tensor(0.0, device=mu.device)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

def classification_loss(logits, target_index, mu, logvar, beta=1.0):
    """
    logits: [1, num_classes]
    target_index: int (masked ingredient index)
    """
    ce_loss = F.cross_entropy(logits, target_index.unsqueeze(0))  # label shape [1]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = ce_loss + beta * kl_loss
    return total_loss, ce_loss, kl_loss

