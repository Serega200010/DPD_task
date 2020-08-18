# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:32:10 2020

@author: Sereg
"""


from blocks import *
import numpy as np
import torch
import torch.nn as nn
from model import model, train

device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'

x_compl = np.load('data/linear_lte_2c_data1_in.npy')
y_compl = np.load('data/linear_lte_2c_data1_out.npy')

x_real = np.array([[float(compl.real), float(compl.imag)] for compl in x_compl])
y_real = np.array([[float(compl.real), float(compl.imag)] for compl in y_compl])
d = y_real - x_real

'''
x_real  = torch.FloatTensor(x_real).view(1,-1,2).to(device)
y_real  = torch.FloatTensor(y_real).view(1,-1,2).to(device)
'''
print(x_real.shape)
print(y_real.shape)

M = 15
Batch_size = 1000

#X.shape == 153 x 1000 x 15 x 2 == 153 x Batch_size x M x 2
X =[[gen_bat(x_real, M, j) for j in range(i,i+ Batch_size)] 
    for i in range(0, x_real.shape[0], Batch_size )]
X = X[:-1]
X = torch.FloatTensor(X)


#must be 153 x 1000 x 1 x 2
Y = np.array([[y_real[j] for j in range(i,i + Batch_size)] 
    for i in range(0, y_real.shape[0] - Batch_size, Batch_size )]).reshape(X.shape[0],Batch_size,-1,2)
Y = torch.FloatTensor(Y)



Model = model(M)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-2)

Model = train(Model,X,Y, criterion,optimizer, 300,10 )
PATH = 'model.txt'
torch.save(Model.state_dict(), PATH)

predictions = []

for x in X:
    preds = Model(x).view(Batch_size,2)
    for p in preds:
        predictions.append(list(p))
    
for i in range(len(predictions)):
    predictions[i] = complex(predictions[i][0].item(),predictions[i][1
    ].item())
    
d = y_compl[:len(predictions)] - predictions

plt.figure()
plt.psd(y_compl,2048)
plt.psd(predictions,2048)
plt.psd(d,2048)


E = torch.FloatTensor([[comp.real, comp.imag] for comp in d])

print(NMSE(X[:E.shape[0]], E))
