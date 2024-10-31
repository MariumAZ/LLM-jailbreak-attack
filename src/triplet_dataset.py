import ast
import pandas as pd
import torch 
import torch.nn as nn
import random

def map_class_embeddings(df):
    """
    Maps embeddings to a category
    Args:
        df (pd.DataFrame): Training data
    """
    data_dict = (
    df.assign(
        embedding=df["embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    )
    .groupby("category")
    ["embedding"]
    .apply(list)
    .to_dict()
)
    return data_dict

from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Construct a class that yields anchor, positive, negative 
    to train the embeddings
    Args:
        Dataset:
    """
    def __init__(self, data_dict, verbose=False):
        self.data_dict = data_dict
        self.classes = list(self.data_dict.keys())
        self.verbose= False
    def __getitem__(self):
        # select a random anchor:
        class_anchor = random.choice(self.classes)
        # print("class_anchor", class_anchor)
        landmark_anchor = random.choice(self.data_dict[class_anchor])
        # select a positive sample from the same class:
        # make sure it is different than the anchor
        pos_sample = random.choice(self.data_dict[class_anchor])
        while pos_sample == landmark_anchor:
            pos_sample = random.choice(self.data_dict[class_anchor])

        # select a negative sample:
        class_neg =  random.choice(list(set(self.classes) - {class_anchor}))
        # print("class_neg", class_neg)
        neg_sample = random.choice(self.data_dict[class_neg])
    
        # return the triplets: anchor, pos, neg
        return (
            torch.Tensor(landmark_anchor),
            torch.Tensor(pos_sample),
            torch.Tensor(neg_sample),
        )

    
    def __len__(self):
        return sum(len(self.data_dict[c]) for c in self.classes)