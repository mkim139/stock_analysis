import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


import model
import misc

def train(train,val,test,device):

    learning_rate = 0.0001
    batch_size = 32
    epochs = 300

    gru_model = model.model()
    optimizer = optim.Adam(lr=learning_rate,weight_decay=0.001)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)

    traindataset = misc.customdataset(trainX,trainY)
    valdataset = misc.customdataset(valX,valY)
    testdataset = misc.customdataset(testX,testY)

    traindataloader = DataLoader(traindataset,batch_size=batch_size,num_workers=0,shuffle=True)
    valdataloader = DataLoader(valdataset,batch_size=batch_size,num_workers=0,shuffle=False)
    testdataloader = DataLoader(testdataset,batch_size=batch_size,num_workers=0,shuffle=False)

    for _ in range(300):
        for bi, (seq,target) in enumerate(traindataloader):
            optimizer.zero_grad()
            pred = gru_model(seq.to(device))


            optimizer.step()
        scheduler.step()

    return