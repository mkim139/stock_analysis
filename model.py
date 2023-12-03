import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self,):
        super(model, self).__init__()

        self.gru = nn.GRU(2,64,num_layers=2)
        self.pred = nn.Linear(128,1)

    def forward(self, x):
        _,out = self.gru(x)
        out = out.permute(1,0,2).flatten(1)
        out = self.pred(out)
        return out.flatten()
        