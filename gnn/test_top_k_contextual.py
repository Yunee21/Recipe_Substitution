from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from lib import utils as uts


def get_top_k(emb_dct, target_name, k=5):
    names = list(emb_dct.keys())
    vectors = np.array([emb_dct[n] for n in names])

    target_vector = emb_dct[target_name].reshape(1,-1)
    sims = cosine_similarity(target_vector, vectors).flatten()

    topk_indices = sims.argsort()[-(k+1):][::-1]  # exclude self
    topk_names = [names[i] for i in topk_indices if names[i] != target_name][:k]

    return topk_names

def compare_sbert_vs_gnn(sbert_emb: dict, gnn_emb: dict, ingre_lst: list, k=5):
    #print(f"{'Ingredient':<20} | {'SBERT Top-K':<40} | {'GNN Top-K'}")
    #print("-" * 90)

    output_df = pd.DataFrame([], columns=['Target', 'SentenceBERT', 'GNN'])

    output_df['Target'] = ingre_lst

    for idx, ing in enumerate(ingre_lst):
        sbert_topk = get_top_k(sbert_emb, ing, k)
        gnn_topk = get_top_k(gnn_emb, ing, k)

        output_df.at[idx, 'SentenceBERT'] = sbert_topk
        output_df.at[idx, 'GNN'] = gnn_topk

        #print(f"{ing:<20} | {', '.join(sbert_topk):<40} | {', '.join(gnn_topk)}")

    print(output_df.head(len(ingre_lst)).to_string())

def main():

    ingre_node_dct = uts.loadPickle("data/sbert_ingre_node_dct.pkl")
    ingre_name_lst = ingre_node_dct['name']
    ingre_fv_lst   = ingre_node_dct['fv']

    sbert_emb_dct = {name: emb for name, emb in zip(ingre_name_lst, ingre_fv_lst)}

    gnn_emb_dct = uts.loadPickle('results/context_ingre_emb_dct.pkl')


    ingredients_to_test = []
    ingredients_to_test = ['chicken', 'potato']
    for i in [7,21,3,24,12,13,127,324,721,522,130]:
        ingredients_to_test.append(ingre_name_lst[i])

    compare_sbert_vs_gnn(sbert_emb_dct, gnn_emb_dct, ingredients_to_test, k=3)


if __name__ == "__main__":
    main()
