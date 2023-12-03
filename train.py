import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import model
import misc

def train(trainprice,trainvolume,valprice,valvolume,testprice,testvolume,learning_rate,device):
    
    checkpointdir = './model_weight/'
    model_name = 'gru_model.pth'
    batch_size = 32
    epochs = 1000
    tolerance = 50
    init_tolerance = tolerance
    ssp = StandardScaler()
    ssv = StandardScaler()
    gru_model = model.model()
    gru_model.to(device)
    optimizer = optim.Adam(gru_model.parameters(),lr=learning_rate,weight_decay=0.001)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.95 ** epoch)
    
    trainprice = pd.DataFrame(ssp.fit_transform(trainprice))
    trainvolume = pd.DataFrame(ssv.fit_transform(trainvolume))

    valprice = pd.DataFrame(ssp.transform(valprice))
    testprice = pd.DataFrame(ssp.transform(testprice))
    valvolume = pd.DataFrame(ssv.transform(valvolume))
    testvolume = pd.DataFrame(ssv.transform(testvolume))

    trainX = np.transpose(np.stack([np.array(trainprice.iloc[:,:3]),np.array(trainvolume.iloc[:,:3])]),(2,1,0))
    trainY = np.array(trainprice.iloc[:,3])
    valX = np.transpose(np.stack([np.array(valprice.iloc[:,:3]),np.array(valvolume.iloc[:,:3])]),(2,1,0))
    valY = np.array(valprice.iloc[:,3])
    testX = np.transpose(np.stack([np.array(testprice.iloc[:,:3]),np.array(testvolume.iloc[:,:3])]),(2,1,0))
    testY = np.array(testprice.iloc[:,3])

    traindataset = misc.customdataset(trainX,trainY)
    valdataset = misc.customdataset(valX,valY)
    testdataset = misc.customdataset(testX,testY)

    traindataloader = DataLoader(traindataset,batch_size=batch_size,num_workers=0,shuffle=True)
    valdataloader = DataLoader(valdataset,batch_size=batch_size,num_workers=0,shuffle=False)
    testdataloader = DataLoader(testdataset,batch_size=batch_size,num_workers=0,shuffle=False)

    for epoch in range(epochs):
        trainloss = []
        for bi, (seq,target) in enumerate(traindataloader):
            optimizer.zero_grad()
            
            pred = gru_model(seq.permute(1,0,2).to(device))
            loss = loss_fn(pred.flatten(),target.to(device))

            loss.backward()
            optimizer.step()
            trainloss.append(loss.item())
        scheduler.step()
        trainloss = np.mean(trainloss)

        globalvalloss = 1e10000
        valpreds = []
        valloss = []
        for bi, (seq,target) in enumerate(valdataloader):
 
            pred = gru_model(seq.permute(1,0,2).to(device))
            loss = loss_fn(pred.flatten(),target.to(device))
            valloss.append(loss.item())
        valpreds += pred.detach().cpu().numpy().flatten().tolist()
        valloss = np.mean(valloss)

        if (epoch % 50) == 0:
            print('Epoch:',epoch,'validation loss:',round(valloss,3))
        if valloss < globalvalloss:
            torch.save(gru_model.state_dict(),checkpointdir+model_name)
            globalvalloss = valloss
            tolerance = init_tolerance
        else:
            tolerance -= 1
            
        if tolerance==0:
            break

    gru_model.load_state_dict(torch.load(checkpointdir+model_name,map_location=device))
    testpreds = []
    testloss = []
    for bi, (seq,target) in enumerate(testdataloader):
        pred = gru_model(seq.permute(1,0,2).to(device))
        loss = loss_fn(pred.flatten(),target.to(device))
        testloss.append(loss.item())
    testloss = np.mean(testloss)
    testpreds += pred.detach().cpu().numpy().flatten().tolist()
    print('Epoch:',epoch,'test loss:',round(testloss,3))

    return (valY,valpreds), (testY,testpreds)