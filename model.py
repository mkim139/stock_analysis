import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self,):
        super(model, self).__init__()

        self.gru = nn.GRU(2,1)

    def forward(self, x):
        _,out = self.gru(x)
        return out
        