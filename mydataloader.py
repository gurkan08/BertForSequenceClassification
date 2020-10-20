
import torch
from params import Params

class MyDataloader(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], torch.tensor(Params.label2id[self.y[item]]).long()