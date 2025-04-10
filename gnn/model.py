import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

class HeteroEncoderHAN(nn.Module):
    def __init__(self, ingre_feat_dim, direc_feat_dim, hidden_dim, latent_dim, cond_dim, metadata, heads=1, dropout=0.0):
        """
        Args:
            - ingre_feat_dim: ingredient 노드 입력 feature 차원 (예: 16)
            - direc_feat_dim: direction 노드 입력 feature 차원 (예: 8)
            - hidden_dim:     HANConv의 출력 차원 (모든 노드에 대해 동일한 차원, 예: 32)
            - latent_dim:     잠재 공간 차원
            - cond_dim:       조건 벡터 차원 (예: nutrition_label 차원, 8)
            - metadata:       (node_types, edge_types) tuple
                              (['ingredient', 'direction'],
                               ['ingredient', 'co_occurs_with', 'direction'],
                               ['ingredient', 'used_in', 'direction'],
                               ['direction', 'contains', 'ingredient'],
                               ['direction', 'pairs_with', 'direction'],
                               ['direction', 'follows', 'direction'])
            - heads:          HANConv의 attention head 수 (기본 1)
            - dropout:        dropout ratio
        """
        super().__init__()

        # in_channels를 dictionary로 설정
        in_channels = {'ingredient': ingre_feat_dim, 'direction': direc_feat_dim}

        if (hidden_dim % heads != 0):
            raise RuntimeError("hidden_dim을 heads로 나누어 떨어지게 설정해야 합니다.")

        self.han_conv = HANConv(in_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        # 최종 graph-level representation은 ingredient와 direction의 pooled 벡터와 condition vector를 결합
        # 총 차원: 2 * hidden_dim + cond_dim
        self.fc_mu = nn.Linear(2 * hidden_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim + cond_dim, latent_dim)

    def forward(self, data: HeteroData, cond: torch.Tensor):
        # x_dict: 각 노드 타입별 입력 feature (dictionary)
        x_dict = {
                'ingredient': data['ingredient'].x,
                'direction': data['direction'].x
        }

        # edge_index_dict: metadata에 포함된 각 관계에 대해 edge_index를 dictionary로 구성
        edge_index_dict = {}
        for key in data.edge_types:
            edge_index_dict[key] = data[key].edge_index

        # HANConv: 입력 dictionary와 edge_index dictionary를 받아 각 노드 타입의 임베딩 출력 (dictionary)
        x_out = self.han_conv(x_dict, edge_index_dict)

        # Global mean pooling: 각 노드 타입별로 임베딩 벡터들의 평균 계산
        ingre_pool = x_out['ingredient'].mean(dim=0, keepdim=True)   # [1, hidden_dim]
        direc_pool = x_out['direction'].mean(dim=0, keepdim=True)    # [1, hidden_dim]

        # Condition vector: [cond_dim] -> [1, cond_dim]
        cond = cond.unsqueeze(0)

        # Concatenate pooled embeddings와 condition vector: [1, 2 * hidden_dim + cond_dim]
        graph_repr = torch.cat([ingre_pool, direc_pool, cond], dim=1)

        mu = self.fc_mu(graph_repr)          # [1, latent_dim]
        logvar = self.fc_logvar(graph_repr)  # [1, latent_dim]
        return mu, logvar


class HeteroDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, ingre_out_dim, num_of_ingre):
        """
        Args:
            - latent_dim: 잠재 공간 차원 (인코더의 z 차원)
            - cond_dim: 조건 벡터 차원   (예: nutrition_label 차원)
            - hidden_dim: 중간 hidden 차원
            - ingre_out_dim: 재료 노드 출력 feature 차원 (입력과 동일, 예: 16)
            - num_of_ingre: 복원할 ingredient 노드의 총 개수 (패딩을 고려해 고정)

        Role:
            - 인코더에서 생성한 잠재 벡터(z)와 조건 벡터(c)로부터 ingredient 노드의 feature들을 복원

        Note:
            - 전체 재료 노드를 복원하는 방식:
                - 모델이 전체 레시피(6개 재료)의 재구성 결과를 출력하고,
                  loss는 오직 마스킹된 노드(예: 인덱스 i)에 대해서만 계산하는 경우,
                  decoder의 출력은 6×ingre_out_dim이 되고,
                  이후 마스킹된 노드 부분만 선택하여 비교합니다.
                - 전체적인 구조 학습 우수, 일반화 가능
            - 마스킹된 노드만 복원하는 방식:
                - 모델이 오직 마스킹된 노드만 재구성하도록 설계한 경우,
                  decoder의 출력은 직접 1×ingre_out_dim이 됩니다
                - 계산 효율성, 주변 노드들 간의 관계나 전체적인 구조 반영 어려움
        """
        super().__init__()
        self.num_of_ingre = num_of_ingre
        self.ingre_out_dim = ingre_out_dim

        # MLP
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, num_of_ingre * ingre_out_dim)
        self.fc2 = nn.Linear(hidden_dim, ingre_out_dim)

        self.fc3 =  nn.Linear(latent_dim, hidden_dim)

    def forward(self, z, cond):
        # z: [batch, latent_dim] (여기서는 보통 batch=1)

        # 조건 벡터를 배치 차원으로 확장
        cond = cond.unsqueeze(0)  # [1, cond_dim]

        # latent와 condition 벡터 결합
        z_cond = torch.cat([z, cond], dim=1)  # [1, latent_dim + cond_dim]

        h = F.relu(self.fc1(z_cond))

        out = self.fc2(h)  # [1, num_ingre * ingre_out_dim]
        out = torch.tanh(out)

        h = F.relu(self.fc3(z))
        out = torch.tanh(self.fc2(h))
        return out

        # 재구성된 ingredient feature들을 [num_ingre, ingre_out_dim]로 reshape
        #recon_ingre = out.view(self.num_of_ingre, self.ingre_out_dim)

        #return recon_ingre


# Conditional Variational Heterogeneous Graph AutoEncoder 모델
class ConditionalHeteroGraphVAE(nn.Module):
    def __init__(self, ingre_feat_dim, direc_feat_dim, hidden_dim, latent_dim, cond_dim, ingre_out_dim, num_of_ingre, metadata, heads=1, dropout=0, num_ingredient_classes=1):
        """
        Args:
            - ingre_feat_dim: ingredient 입력 feature 차원
            - dire_feat_dim:  direction 입력 feature 차원
            - hidden_dim:     내부 hidden 차원
            - latent_dim:     잠재 벡터 차원
            - cond_dim:       조건 벡터 차원
            - ingre_out_dim:  재구성된 ingredient feature 차원 (보통 ingre_in_dim과 동일)
            - num_of_ingre:      각 레시피에 대해 고정된 ingredient 노드 수 (패딩 포함)

        Role:
            Encoder와 Decoder를 하나의 모델로 결합하고,
            reparameterization 기법을 사용하여 z를 샘플링한 후 Decoder로 전달합니다.
        """
        super().__init__()
        self.encoder = HeteroEncoderHAN(ingre_feat_dim, direc_feat_dim, hidden_dim, latent_dim, cond_dim, metadata, heads=heads, dropout=dropout)
        self.decoder = HeteroDecoder(latent_dim, cond_dim, hidden_dim, ingre_out_dim, num_of_ingre)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data: HeteroData):
        # data에서 condition vector는 data.nutrition_label로 제공된다고 가정
        cond = data.nutrition_label          # [cond_dim]

        # compute distribution
        mu, logvar = self.encoder(data, cond)

        # sampling in distribution
        z = self.reparameterize(mu, logvar)  # [1, latent_dim]

        # reconstruction the graph
        #recon_ingre = self.decoder(z, cond)  # [num_ingre, ingre_out_dim]
        recon_ingre = self.decoder(z, cond)  # [1, ingre_out_dim]

        return recon_ingre, mu, logvar
