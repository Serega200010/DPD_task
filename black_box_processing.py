# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:33:17 2020

@author: Sereg
"""


import numpy as np
import torch 
import scipy.io
import scipy
from model import *

mat = scipy.io.loadmat('data/BlackBoxData_80.mat')

x = mat['x'][0]
y = mat['y'][0]

x_real = np.array([[float(compl.real), float(compl.imag)] for compl in x])//2**15
y_real = np.array([[float(compl.real), float(compl.imag)] for compl in y])//2**15

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


predictions = []

for x in X:
    preds = Model(x).view(Batch_size,2)
    for p in preds:
        predictions.append(list(p))

plt.psd(pred_c,2048)
plt.psd(y//2**15,2048)