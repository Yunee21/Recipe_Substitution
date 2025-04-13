import random
import ast
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
import fasttext
import fasttext.util
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import spacy
from lib import utils as uts


def clean_phrase(nlp, text: str) -> str:
    """
    복합 단어(구)를 입력받아 다음을 수행:
    1. 소문자화
    2. 앞뒤 공백 제거
    3. 명사/고유명사 토큰만 추출하여 lemmatize
    4. 결과를 공백으로 다시 join하여 반환
    """
    doc = nlp(text.strip().lower())
    lemmas = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"} and len(token.lemma_)>=2]
    output = " ".join(lemmas)
    output = output.strip()
    return output

def main():
    print("🚀 이 파일이 실행되었습니다:", __file__)

    # *** Recipe Random Sampling ***

    random.seed(721)

    f_name = 'data/recipeNLG-full_dataset.csv'
    recipe_nlg = pd.read_csv(f_name)

    cols = ['title', 'NER', 'directions']

    N = len(recipe_nlg)
    m = 15000

    sampled = random.sample(range(N), m)
    #print(sampled)

    recipe_dct = {}
    for n in tqdm(sampled):
        recipe_title = recipe_nlg.at[n, cols[0]]
        recipe_ingre = recipe_nlg.at[n, cols[1]]
        recipe_direc = recipe_nlg.at[n, cols[2]]

        recipe_ingre = ast.literal_eval(recipe_ingre)
        recipe_direc = ast.literal_eval(recipe_direc)

        recipe_dct[recipe_title] = {'ingredient': recipe_ingre, 'direction': recipe_direc}

    print(len(recipe_dct))

    uts.save2pickle('data/recipe_dct.pkl', recipe_dct)


    # *** Ingredient Node ***

    nlp = spacy.load("en_core_web_sm")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    pca_100d = PCA(n_components=16) # ingre
    pca_50d  = PCA(n_components=32) # direc

    ingre_name_lst = []
    ingre_fv_lst = []
    sbert_ingre_fv_lst = []
    for _, recipe in tqdm(recipe_dct.items()):
        recipe_ingre = recipe['ingredient']
        for ingre in recipe_ingre:
            ingre = clean_phrase(nlp, ingre)

            if (ingre not in ingre_name_lst) and (len(ingre) >= 3):
                ingre_name_lst.append(ingre)

                # SBERT
                emb_384d = model.encode(ingre)
                sbert_ingre_fv_lst.append(emb_384d)

                # FASTTEXT
                emb_300d = ft.get_word_vector(ingre)
                ingre_fv_lst.append(emb_300d)

    print(len(ingre_name_lst))
    print(ingre_fv_lst[-1].shape)
    #print(ingre_name_lst)
    #tmp = list(set(ingre_name_lst))
    #print(len(tmp))
    #return

    ingre_fv_lst = pca_100d.fit_transform(ingre_fv_lst)
    sbert_ingre_fv_lst = pca_100d.fit_transform(sbert_ingre_fv_lst)

    ingre_node_dct = {'name': ingre_name_lst, 'fv': ingre_fv_lst}
    sbert_ingre_node_dct = {'name': ingre_name_lst, 'fv': sbert_ingre_fv_lst}

    uts.save2pickle('data/ingre_node_dct.pkl', ingre_node_dct)
    uts.save2pickle('data/sbert_ingre_node_dct.pkl', sbert_ingre_node_dct)


    # *** Direction Node ***

    #nlp = spacy.load("en_core_web_sm")
    trash = ['.', ',', '/']

    direc_name_lst = []
    direc_fv_lst = []
    for _, recipe in tqdm(recipe_dct.items()):
        recipe_direc = recipe['direction']
        for direc in recipe_direc:
            doc = nlp(direc)  # direc is one of steps
            for token in doc:
                if token.pos_ == "VERB":
                    direc_verb = token.lemma_.lower()

                    if (direc_verb not in direc_name_lst) and (len(direc_verb) >= 3) and (len(direc_verb) < 13):

                        _sum = 0
                        for t in trash:
                            if t in direc_verb:
                                _sum += 1
                        if (_sum != 0):
                            continue

                        direc_name_lst.append(direc_verb)

                        # SBERT
                        #emb_384d = model.encode(direc_verb)
                        #direc_fv_lst.append(emb_384d)

                        # FASTTEXT
                        emb_300d = ft.get_word_vector(ingre)
                        direc_fv_lst.append(emb_300d)

    print(len(direc_name_lst))
    print(direc_fv_lst[-1].shape)
    #print(direc_name_lst)
    #return

    direc_fv_lst = pca_50d.fit_transform(direc_fv_lst)

    direc_node_dct = {'name': direc_name_lst, 'fv': direc_fv_lst}

    uts.save2pickle('data/direc_node_dct.pkl', direc_node_dct)


    # *** Relations ***

    #recipe_dct = uts.loadPickle("data/recipe_dct.pkl")
    #ingre_node_dct = uts.loadPickle("data/ingre_node_dct.pkl")
    #direc_node_dct = uts.loadPickle("data/direc_node_dct.pkl")

    nlp = spacy.load("en_core_web_sm")

    recipe_graph_dct = {}
    for r_title, recipe in tqdm(recipe_dct.items()):

        # (1) co-occurs with (i-i) (undirected)
        src = []
        des = []
        recipe_ingre = recipe['ingredient']
        recipe_ingre = [clean_phrase(nlp,ingre) for ingre in recipe_ingre]
        recipe_ingre = [ingre for ingre in recipe_ingre if len(ingre)>=3]
        for i1 in recipe_ingre:
            for i2 in recipe_ingre:
                if (i1 != i2):
                    src.append(uts.node2id(i1, recipe_ingre))
                    des.append(uts.node2id(i2, recipe_ingre))
        co_occurs_with_lst = [src,des]
        if len(co_occurs_with_lst) == 0 :
            continue

        # (2) short_direc = [[(d,i),(d,i)], [(d,i),(d,i)], ..., []]
        recipe_direc = recipe['direction']
        short_direc = []
        for direc in recipe_direc:
            doc = nlp(direc)
            verb_obj_pairs = []
            for token in doc:
                if token.pos_ == "VERB":
                    direc_verb = token.lemma_.lower()  # this is direc node

                    if len(direc_verb)<3 or len(direc_verb)>=13:
                        continue

                    _sum = 0
                    for t in trash:
                        if t in direc_verb:
                            _sum += 1
                    if _sum != 0:
                        continue

                    for child in token.children:
                        if child.dep_ == "dobj":
                            ingre_raw = child.text
                            ingre_lem = child.lemma_
                            ingre_obj = clean_phrase(nlp, child.text)

                            find = False
                            for ingre in recipe_ingre:
                                doc2 = nlp(ingre)
                                for token2 in doc2:
                                    if (ingre_obj in token2.text) or (ingre_raw in token2.text) or (ingre_lem in token2.text):
                                        verb_obj_pairs.append((direc_verb, ingre))
                                        find = True
                                        break
                                if find:
                                    break
            if (verb_obj_pairs):
                short_direc.append(verb_obj_pairs)

        #print(short_direc)
        recipe_direc = [d for short_step in short_direc for d, _ in short_step]
        recipe_direc = list(set(recipe_direc))

        # (3) used-in (i->d) / contains (d->i) (directed)
        src = []
        des = []
        for short_step in short_direc:
            for pair in short_step:
                d, i = pair
                d_id = uts.node2id(d, recipe_direc)
                i_id = uts.node2id(i, recipe_ingre)
                src.append(d_id)
                des.append(i_id)

        if (len(src) == 0):
            continue
        else:
            contains_lst = [src,des]
            used_in_lst = [des,src]

        # (4) pairs_with (d<->d) / follows (d->d)
        src = []
        des = []
        for short_step in short_direc:
            for d1, _ in short_step:
                for d2, _ in short_step:
                    if d1 != d2 :
                        d1_id = uts.node2id(d1, recipe_direc)
                        d2_id = uts.node2id(d2, recipe_direc)
                        src.append(d1_id)
                        des.append(d2_id)
        pairs_with_lst = [src,des]

        src = []
        des = []
        if len(short_direc) > 1:
            for idx in range(len(short_direc)-1):
                short_step1 = short_direc[idx]
                short_step2 = short_direc[idx+1]
                for d1, _ in short_step1:
                    for d2, _ in short_step2:
                        d1_id = uts.node2id(d1, recipe_direc)
                        d2_id = uts.node2id(d2, recipe_direc)
                        src.append(d1_id)
                        des.append(d2_id)
        follows_lst = [src,des]

        recipe_graph_dct[r_title] = {
            'ingredient': recipe_ingre,
            'direction': recipe_direc,
            'co_occurs_with': co_occurs_with_lst,
            'used_in': used_in_lst,
            'contains': contains_lst,
            'pairs_with': pairs_with_lst,
            'follows': follows_lst,
        }

    print(recipe_graph_dct)
    print(f"총 {len(recipe_graph_dct)}개의 레시피가 합계되었습니다.")

    uts.save2pickle("data/recipe_graph_dct.pkl", recipe_graph_dct)

if __name__ == "__main__":
    main()
