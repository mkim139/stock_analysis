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
    
class model_attention(nn.Module):
    def __init__(self,):
        super(model_attention, self).__init__()

        self.gru = nn.GRU(2,64,num_layers=2)
        self.attention = nn.Sequential(
            nn.Linear(128,64,bias=False),
            nn.Tanh(),
            nn.Linear(64,2,bias=False)
        )
        self.pred = nn.Linear(128,1)

    def forward(self, x):
        _,out = self.gru(x)
        att = self.attention(out)
        att = F.softmax(att,dim=0).permute(1,0,2)

        AA = torch.bmm(att.permute(0,2,1),att)
        I = torch.eye(2)
        penalty = torch.norm(AA-I.to('cuda'))**2
        out = torch.bmm(out.permute(1,2,0),att).sum(2).flatten(1)
        out = self.pred(out)

        return out.flatten(), penalty
        