import random
import torch
import numpy as np
import pandas as pd
import pickle
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

def makeCoOccursWith(ingredients):
    src = []
    des = []
    for i1 in ingredients:
        for i2 in ingredients:
            if (i1 != i2):
                src.append(node2id(i1, ingredients))
                des.append(node2id(i2, ingredients))

    co_occurs_with = [src, des]
    return co_occurs_with
    
def calBMI(weight, height):
    return float(weight) / (float(height)*float(height))

def getNutLabels(disease_info: str):
    disease_lst = ['3단계', '4단계', '혈액투석']
    #assert(disease_info in disease_lst)

    #print("BMI 정상 가정")
    #print(f"{disease_lst}에서 {disease_info} 입력됨")

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
