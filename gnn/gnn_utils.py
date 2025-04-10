import random
import  torch
from torch_geometric.data import HeteroData

def maskIngredientNode(data: HeteroData):
    """
    Args:
        - data: 단일 레시피 HeteroData 객체.
    Returns:
        - mask_id, 원래의 feature vector, 수정된 data
    Roles:
        - ingredient 노드 중 하나를 무작위로 마스킹(해당 노드 feature를 0으로 설정)
          그리고 마스킹 정보를 data['ingredient'].mask에 boolean 텐서로 저장.
    """
    num_of_ingre = data['ingredient'].x.size(0)
    mask_id = random.randint(0, num_of_ingre - 1)
    orig_feature = data['ingredient'].x[mask_id].clone()

    # masking: set masked feature vector to zero
    data['ingredient'].x[mask_id] = torch.zeros_like(orig_feature)

    # save the masking information (shape: [num_of_ingre])
    data['ingredient'].mask = torch.zeros(num_of_ingre, dtype=torch.bool)
    data['ingredient'].mask[mask_id] = True

    return mask_id, orig_feature, data


def padIngredientNode(data: HeteroData, max_num_of_ingre: int):
    '''
    Args:
        - data: 단일 레시피 HeteroData 객체.
        - max_num_of_ingre: 해당 배치 또는 데이터셋의 최대 재료 노드 수
    Roles:
        - Since the recipes have different num of ingredients, the GNN model can not compute.
          So we have to pad.
    '''
    device = data['ingredient'].x.device

    num_of_ingre = data['ingredient'].x.size(0)
    if (num_of_ingre < max_num_of_ingre):
        pad_size = max_num_of_ingre - num_of_ingre

        # padding
        pad_feat = torch.zeros(pad_size, data['ingredient'].x.size(1), device=device)
        data['ingredient'].x = torch.cat([data['ingredient'].x, pad_feat], dim=0)

        # If Ingredient has already masking then, just padding
        if 'mask' in data['ingredient']:
            pad_mask = torch.zeros(pad_size, dtype=torch.bool, device=device)
            data['ingredient'].mask = torch.cat([data['ingredient'].mask.to(device), pad_mask], dim=0)
        else:
            raise KeyError("There are no masking in ingredient node")

    elif (num_of_ingre > max_num_of_ingre):
        raise ValueError("num_of_ingre can not exceed to max_num_of_ingre")

    return data


def generateMaskedVariants(data: HeteroData):
    """
    한 레시피의 모든 ingredient 노드 각각을 마스킹한 여러 HeteroData 샘플 생성
    Returns: List of (masked_id, masked_name, masked_data)
    """
    variants = []
    num_ingre = data['ingredient'].x.size(0)
    for i in range(num_ingre):
        variant = data.clone()
        orig_feat = variant['ingredient'].x[i].clone()
        variant['ingredient'].x[i] = torch.zeros_like(orig_feat)
        variant['ingredient'].mask = torch.zeros(num_ingre, dtype=torch.bool)
        variant['ingredient'].mask[i] = True
        masked_name = variant['ingredient'].name[0][i]

        variants.append((i, masked_name, orig_feat, variant))
    return variants

