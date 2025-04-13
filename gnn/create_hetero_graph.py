import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, DataLoader
from collections import defaultdict
from lib import utils as uts


def analyze_hetero_graphs(graphs):
    stats = defaultdict(list)

    for idx, data in enumerate(graphs):
        graph_stat = {}
        node_counts = {ntype: data[ntype].num_nodes for ntype in data.node_types}
        edge_counts = {rel: data[rel].edge_index.size(1) for rel in data.edge_types}

        total_nodes = sum(node_counts.values())
        total_edges = sum(edge_counts.values())

        graph_stat['total_nodes'] = total_nodes
        graph_stat['total_edges'] = total_edges

        for ntype in data.node_types:
            graph_stat[f'{ntype}_nodes'] = node_counts[ntype]

        for etype in data.edge_types:
            graph_stat[f'{etype}_edges'] = edge_counts[etype]

            src_nodes = data[etype].edge_index[0]
            tgt_nodes = data[etype].edge_index[1]

            degree_src = torch.bincount(src_nodes, minlength=node_counts[etype[0]])
            degree_tgt = torch.bincount(tgt_nodes, minlength=node_counts[etype[2]])

            isolated_src = (degree_src == 0).sum().item()
            isolated_tgt = (degree_tgt == 0).sum().item()

            graph_stat[f'{etype}_isolated_src'] = isolated_src
            graph_stat[f'{etype}_isolated_tgt'] = isolated_tgt

        # Degree 평균
        graph_stat['avg_degree'] = total_edges / total_nodes if total_nodes > 0 else 0


        # 밀도 (단순 근사치로: 단방향, 이질성 무시)
        max_possible_edges = total_nodes * (total_nodes - 1)
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        graph_stat['density'] = density

        stats[idx] = graph_stat

    return stats


def to_edge_index(edge_list, device):
    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        return torch.tensor(edge_list, dtype=torch.long, device=device)

def create_heterogeneous_graph(recipe_graph, ingre_node_dct, direc_node_dct, device) :
    data = HeteroData()

    # Ingredient
    recipe_ingre = recipe_graph['ingredient']
    num_of_ingre = len(recipe_ingre)
    ingre_fv_dim = ingre_node_dct['fv'].shape[-1]
    X = torch.zeros((num_of_ingre, ingre_fv_dim))

    for i, ingre_name in enumerate(recipe_ingre) :
        ingre_glob_id = uts.node2id(ingre_name, ingre_node_dct['name'])
        fv = ingre_node_dct['fv'][ingre_glob_id]
        X[i] = torch.tensor(fv)

    data['ingredient'].x = F.normalize(X, dim=-1).to(device)
    data['ingredient'].x = X.to(device)
    data['ingredient'].name = recipe_ingre
    data['ingredient'].node_id = torch.arange(num_of_ingre, device=device)


    # Direction
    recipe_direc = recipe_graph['direction']
    num_of_direc = len(recipe_direc)
    direc_fv_dim = direc_node_dct['fv'].shape[-1]
    X = torch.zeros((num_of_direc, direc_fv_dim))

    for i, direc_name in enumerate(recipe_direc) :
        direc_glob_id = uts.node2id(direc_name, direc_node_dct['name'])
        fv = direc_node_dct['fv'][direc_glob_id]
        X[i] = torch.tensor(fv)

    data['direction'].x = F.normalize(X, dim=-1).to(device)
    data['direction'].x = X.to(device)
    data['direction'].name = recipe_direc
    data['direction'].node_id = torch.arange(num_of_direc, device=device)


    # Relation
    data['ingredient', 'co_occurs_with', 'ingredient'].edge_index = to_edge_index(recipe_graph['co_occurs_with'], device=device)
    data['ingredient', 'used_in', 'direction'].edge_index = to_edge_index(recipe_graph['used_in'], device=device)
    data['direction', 'contains', 'ingredient'].edge_index = to_edge_index(recipe_graph['contains'], device=device)
    data['direction', 'pairs_with', 'direction'].edge_index = to_edge_index(recipe_graph['pairs_with'], device=device)
    data['direction', 'follows', 'direction'].edge_index = to_edge_index(recipe_graph['follows'], device=device)

    return data


def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    seed = 721
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    recipe_graph_dct = uts.loadPickle('data/recipe_graph_dct.pkl')
    ingre_node_dct   = uts.loadPickle('data/ingre_node_dct.pkl')
    direc_node_dct   = uts.loadPickle('data/direc_node_dct.pkl')

    recipe_graphs = []
    for _, recipe_graph in recipe_graph_dct.items():
        if len(recipe_graph['ingredient']) < 3 or len(recipe_graph['direction']) == 0:
            continue
        else:
            recipe_graphs.append(create_heterogeneous_graph(recipe_graph, ingre_node_dct, direc_node_dct, device))

    print(f"# of useful recipes: {len(recipe_graphs)}")
    print(recipe_graphs[0])

    recipe_stats = analyze_hetero_graphs(recipe_graphs)
    for idx in [7, 21, 324]:
        print(f"'[{idx}]' Graph Analysis: ")
        for key, val in recipe_stats[idx].items():
            print(f"{key}: {val}")

    uts.save2pickle('data/heterographs.pkl', recipe_graphs)


if __name__ == "__main__":
    main()
