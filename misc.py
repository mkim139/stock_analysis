import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class customdataset(Dataset):
    def __init__(self, seq,target):
        self.seq = seq
        self.target = target
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        sequence = self.seq[:,idx,:]
        target = self.target[idx]
        return torch.tensor(sequence).float(),torch.tensor(target).long()