import random
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import HeteroData, DataLoader
from torch_geometric.nn import HANConv

from tqdm import tqdm

from lib import utils as uts
from gnn.gnn_utils import maskIngredientNode, padIngredientNode, generateMaskedVariants
from gnn.model import ConditionalHeteroGraphVAE
from gnn.loss import lossFunction, classification_loss
from torch.nn.functional import cosine_embedding_loss


def train(dataloader, max_num_of_ingre, model, optimizer, device, accumulation_steps=1, beta=1.0):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for i, data in enumerate(dataloader):
        data = data.to(device)
        '''
        # original ingredient feature -> target
        target_ingre_feat = data['ingredient'].x.clone()

        pad_size = max_num_of_ingre - target_ingre_feat.size(0)
        pad_feat = torch.zeros(pad_size, target_ingre_feat.size(1), device=device)
        target_ingre_feat = torch.cat([target_ingre_feat, pad_feat], dim=0)

        # random sampling for masking in a recipe
        mask_id, orig_feat, data = maskIngredientNode(data)

        # padding
        data = padIngredientNode(data, max_num_of_ingre)
        '''
        # 각 재료를 마스킹한 레시피 variant 생성
        variants = generateMaskedVariants(data)

        for mask_id, masked_name, orig_feat, variant in variants:
            # padding
            variant = padIngredientNode(variant, max_num_of_ingre)
            data = variant.clone()

            #recon_ingre, mu, logvar = model(data) # regression

            recon, mu, logvar = model(data) # regression
            target = orig_feat.to(device)

            recon = recon.squeeze(0)

            if torch.norm(recon) == 0 or torch.norm(target) == 0:
                print("🚨 Zero vector detected before normalization")


            cosine_loss = F.cosine_embedding_loss(
                recon.unsqueeze(0),      # [1, 100]
                target.unsqueeze(0),     # [1, 100]
                torch.tensor([1.], device=device)  # label: positive pair
            )
            #recon_loss = F.mse_loss(recon, target)
            recon_loss = F.smooth_l1_loss(recon, target, reduction='sum')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            #loss = recon_loss + beta * kl_loss + cosine_loss
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        '''
        # compute loss only for masking node in regression
        _, recon_loss, kl_loss = lossFunction(recon_ingre, target_ingre_feat, mu, logvar, data['ingredient'].mask)

        # Warm-up을 위해 KL 항에 beta를 곱해 새로 정의한 loss
        loss = recon_loss + beta * kl_loss

        # --- Gradient Accumulation 핵심 ---
        # 보통, accumulation 시에는 각 step의 loss를 accumulation_steps로 나누어 backward합니다.
        # 그래야 gradient가 너무 커지지 않고, 실제로 batch_size=4로 처리한 것과 유사해집니다.
        loss = loss / accumulation_steps
        loss.backward()

        # gradient를 몇 번 쌓았는지 세고,
        # 정해진 accumulation_steps에 도달하면 한 번 optimizer.step().
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 통계용
        total_loss += loss.item() * accumulation_steps  # 실제 스케일(곱셈 해제)
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

    # 만약 데이터 개수가 accumulation_steps로 나누어떨어지지 않는 경우,
    # 루프가 끝난 뒤 남아 있는 gradient를 마지막으로 step()할 필요가 있을 수 있음.
    # (기본 예시는 생략, 필요하면 아래처럼 처리)
    remainder = len(dataloader) % accumulation_steps
    if remainder != 0:
        optimizer.step()
        optimizer.zero_grad()
    '''

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    #print(f"Average Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    return avg_loss, avg_recon_loss, avg_kl_loss

def validate(dataloader, max_num_of_ingre, model, device):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            '''
            target_ingre_feat = data['ingredient'].x.clone()

            pad_size = max_num_of_ingre - target_ingre_feat.size(0)
            pad_feat = torch.zeros(pad_size, target_ingre_feat.size(1), device=device)
            target_ingre_feat = torch.cat([target_ingre_feat, pad_feat], dim=0)

            # also do masking in validation (혹은 masking 없이 전체 재구성을 평가할 수도 있음)
            mask_id, orig_feat, data = maskIngredientNode(data)

            # padding
            data = padIngredientNode(data, max_num_of_ingre)
            '''
            variants = generateMaskedVariants(data)

            for mask_id, masked_name, orig_feat, variant in variants:
                data = variant.clone()

                recon, mu, logvar = model(data) # regression
                target = orig_feat.to(device)
                recon = recon.squeeze(0)

                cosine_loss = F.cosine_embedding_loss(
                    recon.unsqueeze(0),      # [1, 100]
                    target.unsqueeze(0),     # [1, 100]
                    torch.tensor([1.], device=device)  # label: positive pair
                )
                #recon_loss = F.mse_loss(recon, target)
                recon_loss = F.smooth_l1_loss(recon, target, reduction='sum')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss + cosine_loss


                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                '''

                recon_ingre, mu, logvar = model(data)

                loss, recon_loss, kl_loss = lossFunction(recon_ingre, target_ingre_feat, mu, logvar, data['ingredient'].mask)
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                '''

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    return avg_loss, avg_recon_loss, avg_kl_loss


def run(recipe_graphs, hidden_dim=64, latent_dim=32, heads=1, dropout=0.0, num_epochs=20, batch_size=1, accumulation_steps=4, lr=1e-3):

    device = torch_geometric.device('auto')
    print(f"Device: {device}")

    # 기존 recipe_graphs 리스트에서 direction 노드가 하나라도 있는 데이터만 필터링 (loss = nan 을 방지
    filtered_recipe_graphs = [data for data in recipe_graphs if data['direction'].x.size(0) > 0]
    assert len(recipe_graphs) == len(filtered_recipe_graphs)

    # *** Padding ***
    max_num_of_ingre = max([data['ingredient'].x.size(0) for data in recipe_graphs])
    print(f"모든 레시피 중에서 ingredient 개수가 가장 많을 때는 {max_num_of_ingre}개 입니다.")

    #recipe_graphs_with_padding = [padIngredientNode(data, max_num_ingre) for data in recipe_graphs]
    #print(recipe_graphs_with_padding[0])
    #print(recipe_graphs_with_padding[0]['ingredient'].x)

    # 전체 데이터셋 분할 (70% train, 30% validation)
    dataset_size = len(recipe_graphs)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(recipe_graphs, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = recipe_graphs[0].metadata()
    print(f"metadata: {metadata}")

    # hyperparameter
    ingre_feat_dim = recipe_graphs[0]['ingredient'].x.shape[1]
    direc_feat_dim = recipe_graphs[0]['direction'].x.shape[1]
    cond_dim = len(recipe_graphs[0].nutrition_label)
    ingre_out_dim = ingre_feat_dim
    #num_of_ingre = recipe_graphs[0]['ingredient'].x.size(0)  # 각 레시피의 ingredient 수 (패딩된 값)
    num_of_ingre = max_num_of_ingre

    model = ConditionalHeteroGraphVAE(ingre_feat_dim, direc_feat_dim, hidden_dim, latent_dim, cond_dim, ingre_out_dim, num_of_ingre, metadata, heads=heads, dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        beta = min(1.0, (epoch / 10.0))
        train_loss, train_recon_loss, train_kl_loss = train(train_loader, max_num_of_ingre, model, optimizer, device, accumulation_steps, beta)
        val_loss, val_recon_loss, val_kl_loss = validate(val_loader, max_num_of_ingre, model, device)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: [total: {train_loss:.4f}, recon: {train_recon_loss:.4f}, kl: {train_kl_loss:.4f}]")
        print(f"             - Val Loss:   [total: {val_loss:.4f}, recon: {val_recon_loss:.4f}, kl: {val_kl_loss:.4f}]")
        #print(f"Epoch {epoch}/{num_epochs} - Train Loss: [total: [{train_loss:.4f}], recon: [{train_recon_loss:.4f}], kl: [{train_kl_loss:.4f}],
        #                                       Val Loss: [total: [{val_loss:.4f}], recon: [{val_recon_loss:.4f}], kl: [{val_recon_loss:.4f}]")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    print("Training complete. Best validation loss:", best_val_loss)


def main():
    seed = 721
    random.seed(721)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # *** Load data ***
    recipe_graphs = uts.loadPickle("gnn/results/recipe_graphs_lst.pkl")
    #recipe_graphs = DataLoader(recipe_graphs_lst, batch_size=1, shuffle=True)
    '''
    # *** Padding ***
    max_num_ingre = max([data['ingredient'].x.size(0) for data in recipe_graphs])
    print(f"모든 레시피 중에서 ingredient 개수가 가장 많을 때는 {max_num_ingre}개 입니다.")

    recipe_graphs_with_padding = [padIngredientNode(data, max_num_ingre) for data in recipe_graphs]
    print(recipe_graphs_with_padding[0])
    print(recipe_graphs_with_padding[0]['ingredient'].x)
    '''
    run(recipe_graphs,
        hidden_dim=128,
        latent_dim=32,
        heads=4,
        dropout=0.5,
        num_epochs=50,
        batch_size=1,
        accumulation_steps=1,
        lr=1e-6
    )


if __name__ == "__main__":
    main()
