import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HANConv
from torch.nn.functional import cosine_embedding_loss

from gnn.gnn_utils import maskIngredientNode, maskIngredientNodes
from gnn.model import ContextualHeteroGraphVAE
from lib import utils as uts


def loss_function(recon, target, mu, logvar, use_cosine=False, beta=1.0):
    """
    recon: [1, d] - 모델이 복원한 재료 feature
    target: [d] - 원래 feature
    mu, logvar: VAE latent stats
    """
    #recon_loss = F.mse_loss(recon.squeeze(0), target, reduction='sum')

    #recon = F.normalize(recon, dim=-1)
    #target = F.normalize(target, dim=-1)
    #print(recon)
    #print(target)
    recon_loss = F.smooth_l1_loss(recon.squeeze(0), target, reduction='sum')

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    if use_cosine:
        recon = F.normalize(recon, dim=-1)
        target = F.normalize(target, dim=-1)

        #cosine_loss = 1 - F.cosine_similarity(recon, target.unsqueeze(0))
        target_label = torch.ones(recon.size(0), device=recon.device)  # [B]
        cosine_loss = F.cosine_embedding_loss(
            recon, target, target_label
        )
        total_loss = recon_loss + beta * kl_loss + cosine_loss
    else:
        total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss.item(), kl_loss.item()

def train_contextual_vae(model, dataloader, optimizer, device, beta=1.0, use_cosine=False):
    model.train()
    total_loss, total_recon_loss, total_kl_loss = 0, 0, 0
    cos_sim = 0

    for data in tqdm(dataloader, desc="Training"):
        data = data.to(device)

        #data['ingredient'].x = F.normalize(data['ingredient'].x, dim=-1)
        #data['direction'].x = F.normalize(data['direction'].x, dim=-1)

        mask_id, target_fv, masked_data = maskIngredientNodes(data, 10)

        recon, mu, logvar = model(masked_data)
        #print(recon.shape)
        #print(target_fv.shape)
        #print(recon)
        #print(target_fv)
        #print(mu)
        #print(logvar)
        #print("mu mean:", mu.mean().item(), "std:", mu.std().item())

        #target_fv = F.tanh(target_fv)

        loss, recon_loss, kl_loss = loss_function(recon, target_fv.to(device), mu, logvar, use_cosine, beta)

        if (not use_cosine):
            recon = F.normalize(recon, dim=-1)
            target_fv = F.normalize(target_fv, dim=-1)
            cos_sim += F.cosine_similarity(recon, target_fv, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        total_loss += loss.item()
        total_recon_loss += recon_loss
        total_kl_loss += kl_loss

    if (not use_cosine):
        print(f"Cosine Similarity: [{cos_sim/len(dataloader):.4f}]")

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)
    return avg_loss, avg_recon, avg_kl

@torch.no_grad()
def validate_contextual_vae(model, dataloader, device, beta=1.0, use_cosine=False):
    model.eval()
    total_loss, total_recon_loss, total_kl_loss = 0, 0, 0

    for data in tqdm(dataloader, desc="Validation"):
        data = data.to(device)

        #data['ingredient'].x = F.normalize(data['ingredient'].x, dim=-1)
        #data['direction'].x = F.normalize(data['direction'].x, dim=-1)

        mask_id, target_fv, masked_data = maskIngredientNodes(data, 10)

        recon, mu, logvar = model(masked_data)

        #target_fv = F.tanh(target_fv)

        loss, recon_loss, kl_loss = loss_function(recon, target_fv.to(device), mu, logvar, use_cosine, beta)

        total_loss += loss.item()
        total_recon_loss += recon_loss
        total_kl_loss += kl_loss

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)
    return avg_loss, avg_recon, avg_kl

@torch.no_grad()
def extract_ingredient_embeddings(model, dataset, device):
    model.eval()
    ingre_emb = {}

    for data in dataset:
        data = data.to(device)

        x_dct = {
            'ingredient': data['ingredient'].x,
            'direction': data['direction'].x
        }

        edge_index_dct = {k: data[k].edge_index for k in data.edge_types}

        x1 = model.encoder.han1(x_dct, edge_index_dct)
        x2 = model.encoder.han2(x1, edge_index_dct)

        for name, emb in zip(data['ingredient'].name, x2['ingredient']):
            ingre_emb[name] = emb.cpu().numpy()

        '''
        x_dct = model.encoder.han_conv(
            {'ingredient': data['ingredient'].x, 'direction': data['direction'].x},
            {k: data[k].edge_index for k in data.edge_types}
        )

        for name, emb in zip(data['ingredient'].name, x_dct['ingredient']):
            ingre_emb[name] = emb.cpu().numpy()
        '''

    return ingre_emb


def main():
    print("🚀 이 파일이 실행되었습니다:", __file__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    heterographs = uts.loadPickle("data/heterographs.pkl")

    n_total = len(heterographs)
    n_train = int(0.7 * n_total)
    n_val   = n_total - n_train
    batch_size = 64
    print(f"Data Split: [Total: {n_total}, Train: {n_train}, Validation: {n_val}, Batch size: {batch_size}]")

    train_dataset, val_dataset = random_split(heterographs, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = ContextualHeteroGraphVAE(
        ingre_fv_dim=16,
        direc_fv_dim=32,
        hidden_dim=256,
        latent_dim=32,
        out_dim=16,
        metadata=heterographs[0].metadata(),
        heads=2,
        dropout=0.5
        ).to(device)

    use_cosine = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs=100
    early_stopping_patience=10
    save_path = "results/context_best_model.pt"

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1,num_epochs+1):
        beta = min(1.0, epoch/50)  # KL warm-up

        train_loss, train_recon, train_kl = train_contextual_vae(
            model, train_loader, optimizer, device, beta=beta, use_cosine=use_cosine
        )

        val_loss, val_recon, val_kl = validate_contextual_vae(
            model, val_loader, device, beta=1.0, use_cosine=use_cosine
        )

        if (use_cosine):
            train_cosine = train_loss - train_recon - train_kl
            val_cosine = val_loss - val_recon - val_kl

            print(f"[Epoch {epoch}]")
            print(f"  🔧 Train Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Cosine: {train_cosine:.4f}")
            print(f"  🔍 Val   Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Cosine: {val_cosine:.4f}")

        else:
            print(f"[Epoch {epoch}]")
            print(f"  🔧 Train Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
            print(f"  🔍 Val   Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Best model saved!")

        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break

        if epoch > 15:
            early_stopping_patience = 5

    print("Training process is terminated")
    print("Save the embedding space")
    ingre_emb_dct = extract_ingredient_embeddings(model, heterographs, device)
    uts.save2pickle("results/context_ingre_emb_dct.pkl", ingre_emb_dct)



if __name__ == "__main__":
    main()

