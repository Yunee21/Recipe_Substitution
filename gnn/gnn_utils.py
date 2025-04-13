import random
import  torch
from torch_geometric.data import HeteroData


def maskIngredientNode(data: HeteroData):
    num_of_ingre = data['ingredient'].x.size(0)
    mask_id = random.randint(0, num_of_ingre-1)

    orig_fv = data['ingredient'].x[mask_id].clone()
    data['ingredient'].x[mask_id] = torch.zeros_like(orig_fv)

    data['ingredient'].mask = torch.zeros(num_of_ingre, dtype=torch.bool)
    data['ingredient'].mask[mask_id] = True

    return mask_id, orig_fv, data


def maskIngredientNodes(data: HeteroData, num_mask: int = 3):
    # Assume: Ingredients are more than three at least.
    num_of_ingre = data['ingredient'].x.size(0)
    mask_ids = random.sample(range(num_of_ingre), num_mask)

    orig_fvs = data['ingredient'].x[mask_ids].clone()
    for idx in mask_ids:
        data['ingredient'].x[idx] = torch.zeros_like(data['ingredient'].x[idx])

        data['ingredient'].mask = torch.zeros(num_of_ingre, dtype=torch.bool)
        data['ingredient'].mask[mask_ids] = True

    return mask_ids, orig_fvs, data


def maskDirectionNode(data: HeteroData):
    num_of_direc = data['direction'].x.size(0)
    mask_id = random.randint(0, num_of_direc-1)

    orig_fv = data['direction'].x[mask_id].clone()
    data['direction'].x[mask_id] = torch.zeros_like(orig_fv)

    data['direction'].mask = torch.zeros(num_of_direc, dtype=torch.bool)
    data['direction'].mask[mask_id] = True

    return mask_id, orig_fv, data

