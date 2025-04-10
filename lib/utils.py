import random
import torch
import numpy as np
import pandas as pd
import pickle

import torch.nn.functional as F
from torch_geometric.data import HeteroData, DataLoader

from gnn.model import ConditionalHeteroGraphVAE
from gnn.gnn_utils import padIngredientNode
from gnn.loss import lossFunction

from deep_translator import GoogleTranslator


def save2pickle(file_name: str, data):
    assert file_name[-3:] == "pkl"

    with open(file_name, "wb") as f:
        pickle.dump(data, f)

    print(f"Data successfully saved to {file_name}")


def loadPickle(file_name: str):
    assert file_name[-3:] == "pkl"

    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data

def id2node(id: int, data: list) -> str:
    return data[id]

def node2id(node: str, data: list) -> int:
    return data.index(node)

def eng2ko(word: str):
    return GoogleTranslator(source='en', target='ko').translate(word)

def ko2eng(word: str):
    return GoogleTranslator(source='ko', target='en').translate(word)
    
def calBMI(weight, height):
    return float(weight) / (float(height)*float(height))

def getNutLabels(disease_info: str):
    disease_lst = ['3단계', '4단계', '혈액투석']
    assert(disease_info in disease_lst)

    print("BMI 정상 가정")
    print(f"{disease_lst}에서 {disease_info} 입력됨")

    # low fat | low potassium | high protein | low protien | low phosphorus | low sodium | high potassium
    nut_label_vec = []

    if (disease_info in disease_lst[:2]):
        # 제한: 칼륨, 단백질, 인, 나트륨
        nut_label_vec = [0, 1, 0, 1, 1, 1, 0]

    elif (disease_info in disease_lst[2]):
        # 제한: 칼륨, 나트륨 / 적절: 단백질
        nut_label_vec = [0, 1, 0, 0, 0, 1, 0]

    else:
        print("Wrong Disease")

    #print(nut_label_vec)
    return nut_label_vec

def createHeteroGraph(recipe, ingredient_node: dict, direction_node: dict, device):
    data = HeteroData()

    # ingredient
    ingredients = recipe['ingredients']
    num_of_ingre = len(ingredients)
    ingre_featvec_dim = ingredient_node['feature_vector'].shape[-1]
    X = torch.zeros((num_of_ingre, ingre_featvec_dim))
    for idx, ingre_name in enumerate(ingredients):
        ingre_glob_id = uts.node2id(ingre_name, ingredient_node['name'])
        feat_vec = ingredient_node['feature_vector'][ingre_glob_id]
        X[idx] = feat_vec

    data['ingredient'].x = X.to('cuda')
    data['ingredient'].name = ingredients
    data['ingredient'].node_id = torch.arange(num_of_ingre, device=device)

    # direction
    directions = recipe['directions']
    num_of_direc = len(directions)
    direc_featvec_dim = direction_node['feature_vector'].shape[-1]
    X = torch.zeros((num_of_direc, direc_featvec_dim))
    for idx, direc_name in enumerate(directions):
        direc_glob_id = uts.node2id(direc_name, direction_node['name'])
        feat_vec = direction_node['feature_vector'][direc_glob_id]
        X[idx] = feat_vec

    data['direction'].x = X.to('cuda')
    data['direction'].name = directions
    data['direction'].node_id = torch.arange(num_of_direc, device=device)

    # relation
    data['ingredient', 'co_occur_withs', 'ingredient'].edge_index = torch.tensor(recipe['co_occurs_with'], dtype=torch.long, device=device)
    data['ingredient', 'used_in', 'direction'].edge_index = torch.tensor(recipe['used_in'], dtype=torch.long, device=device)
    data['direction', 'contains', 'ingredient'].edge_index = torch.tensor(recipe['contains'], dtype=torch.long, device=device)
    data['direction', 'pairs_with', 'direction'].edge_index = torch.tensor(recipe['pairs_with'], dtype=torch.long, device=device)
    data['direction', 'follows', 'direction'].edge_index = torch.tensor(recipe['follows'], dtype=torch.long, device=device)

    # recipe condition vector
    data.nutrition_label = torch.tensor(recipe['nutrition_label_encodings'], dtype=torch.float, device=device)

    #print(data)
    return data

def maskIngredient(data, mask_indices):
    """
    사용자가 지정한 mask_indices 에 해당하는 ingredient 노드를 마스킹 처리.
    기존 maskIngredientNode 대신, 원하는 노드만 수동으로 마스킹
    """
    num_ingre = data['ingredient'].x.size(0)

    # 먼저 mask를 모두 0으로 초기화
    # mask 속성이 없으면 새로 만든다 (dtype=torch.long 또는 bool 등)
    if not hasattr(data['ingredient'], 'mask') or data['ingredient'].mask is None:
        data['ingredient'].mask = torch.zeros(num_ingre, dtype=torch.long)

    else:
        data['ingredient'].mask[:] = 0

    # mask_indices에 해당하는 위치를 1로 설정
    m_idx = mask_indices
    #for idx in mask_indices:
    data['ingredient'].mask[m_idx] = 1

    if hasattr(data['ingredient'], 'name'):
        m_idx = mask_indices
        #for idx in mask_indices:
        ingre_name = data['ingredient'].name[m_idx]
        print(f"[Masking] 노드 인덱스={m_idx}, 재료명={ingre_name}")

    else:
        print("주의: data['ingredient'].name 속성이 없음. 재료명을 알 수 없습니다.")

    # 실제 x값을 0으로 바꾸거나, 일부 임베딩을 지우는 등의 로직을 원하면:
    # data['ingredient'].x[mask_indices] = 0.0
    # (모델 내부 mask 처리 방식을 따라 조정하면 됩니다.)

    #print(data)
    return data

def get_top_k_similar_ingredients(recon_vector, ingredient_embeddings, ingredient_lst, k=5):
    """
    recon_vector: (feat_dim,)  # 재구성된 단일 노드의 임베딩 벡터
    ingredient_embeddings: (N, feat_dim)  # 라이브러리(사전) 내 ingredient 임베딩들
    ingredient_lst: 길이 N, 각 행이 어떤 ingredient인지 이름을 담고 있다고 가정
    k: 상위 몇 개를 뽑을지

    return: [(ingredient_name, similarity), ...] 형태로 상위 k개
    """
    # (1) 차원 맞추기: recon_vector -> (1, feat_dim)
    recon_vector = recon_vector.unsqueeze(0)  # shape: (1, feat_dim)

    # (2) 코사인 유사도: ingredient_embeddings와 한꺼번에 계산
    #     -> 결과 shape: (N,)
    cos_sim = F.cosine_similarity(recon_vector, ingredient_embeddings, dim=1)  # (N,)

    # (3) topk
    topk_vals, topk_indices = torch.topk(cos_sim, k, dim=0)

    results = []
    for rank in range(k):
        idx = topk_indices[rank].item()
        sim_score = topk_vals[rank].item()
        ingre_name = ingredient_lst[idx]
        results.append((ingre_name, sim_score))

    return results

def inference(
        test_graph,
        model_path="best_model.pt",
        recipe_graph_path="gnn/results/recipe_graphs_lst.pkl",
        ingredient_node=None,
        mask_indices=[0]
    ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # (1) 학습된 모델과 동일한 구조로 모델 준비
    recipe_graphs = uts.loadPickle(recipe_graph_path)
    max_num_of_ingre = max([data['ingredient'].x.size(0) for data in recipe_graphs])

    ingre_feat_dim = test_graph['ingredient'].x.shape[1]
    direc_feat_dim = test_graph['direction'].x.shape[1]
    cond_dim = len(test_graph.nutrition_label)
    metadata = test_graph.metadata()

    model = ConditionalHeteroGraphVAE(
            ingre_feat_dim, direc_feat_dim,
            hidden_dim=128,
            latent_dim=32,
            cond_dim=cond_dim,
            ingre_out_dim=ingre_feat_dim,
            num_of_ingre=max_num_of_ingre,
            metadata=metadata,
            heads=4,
            dropout=0.5
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")

    # (2) 원하는 ingredient를 마스킹
    test_graph = maskIngredient(test_graph, mask_indices[0])
    #print(f"Mask indices: {mask_indices}")

    # (3) 패딩
    test_graph = test_graph.to(device)
    test_graph = padIngredientNode(test_graph, max_num_of_ingre)
    #print(test_graph)

    # (4) Forward
    with torch.no_grad():
        recon_ingre, mu, logvar = model(test_graph)
        #print(recon_ingre, mu, logvar)

    # (5) reconstruction 결과에서, 마스킹된 노드의 재구성 임베딩 추출
    ingredient_embeddings = ingredient_node['feature_vector']
    ingredient_lst = ingredient_node['name']

    m_idx = mask_indices[0]
    #for m_idx in mask_indices:
    recon_vector = recon_ingre[m_idx]  # shape: (ingre_feat_dim,)

    print(f"\n[Masked Node={m_idx}] Reconstructed embedding (first 5 dims):")
    print(recon_vector[:5].cpu().numpy())

    # (6) 코사인 유사도 Top-5
    if ingredient_embeddings is not None and ingredient_lst is not None:
        top_5 = get_top_k_similar_ingredients(
                recon_vector,
                ingredient_embeddings.to(device),
                ingredient_lst,
                k=10
        )

        print("Top-5 similar ingredients:")
        for ingr_name, score in top_5:
            print(f"  {ingr_name} (cos_sim={score:.4f})")

    else:
        print("[주의] ingredient_embeddings / ingredient_lst가 없습니다. "
              "코사인 유사도 Top-5를 계산하려면 이 두 변수가 필요합니다.")
