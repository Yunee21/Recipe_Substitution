import random
import pickle
import numpy as np
import torch
from torch_geometric.data import HeteroData, DataLoader
from lib import utils as uts


def create_heterogeneous_graph(recipe, ingredient_node: dict, direction_node: dict, device):
    data = HeteroData()

    # Ingredient
    ingredients = recipe['lowercase_ingredients']
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

    # Direction
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

    # Relation
    data['ingredient', 'co_occur_withs', 'ingredient'].edge_index = torch.tensor(recipe['co_occurs_with'], dtype=torch.long, device=device)
    data['ingredient', 'used_in', 'direction'].edge_index = torch.tensor(recipe['used_in'], dtype=torch.long, device=device)
    data['direction', 'contains', 'ingredient'].edge_index = torch.tensor(recipe['contains'], dtype=torch.long, device=device)
    data['direction', 'pairs_with', 'direction'].edge_index = torch.tensor(recipe['pairs_with'], dtype=torch.long, device=device)
    data['direction', 'follows', 'direction'].edge_index = torch.tensor(recipe['follows'], dtype=torch.long, device=device)

    # Recipe Condition Vector (Nutrition Label)
    #@Updated date: 25.03.30
    #@brief: (low salt)->(low sodium) [0,0,1(X),0,0,0,1(O),0] \in R^7
    data.nutrition_label = torch.tensor(recipe['nutrition_label_encodings'], dtype=torch.float, device=device)

    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    seed = 721
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    recipe_dct = uts.loadPickle('preprocess/results/recipe_dct.pkl')
    ingredient_node = uts.loadPickle('preprocess/results/ingredient_node.pkl')
    direction_node = uts.loadPickle('preprocess/results/direction_node.pkl')

    recipe_graphs = []
    for _, recipe in recipe_dct.items():
        if (len(recipe) == 0):
            continue
        else:
            recipe_graphs.append(create_heterogeneous_graph(recipe, ingredient_node, direction_node, device))

    print(f"# of useful recipes: {len(recipe_graphs)}")
    print(recipe_graphs[0])

    uts.save2pickle('gnn/results/recipe_graphs_lst.pkl', recipe_graphs)


if __name__ == "__main__":
    main()
