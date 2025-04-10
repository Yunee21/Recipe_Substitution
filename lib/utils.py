import pickle


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

