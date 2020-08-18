# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:36:15 2020

@author: Sereg
"""

import numpy as np
import torch 
import scipy.io
import scipy
from model import *
mat = scipy.io.loadmat('data/BlackBoxData_80.mat')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

x = mat['x'][0]
y = mat['y'][0]

x_real = np.array([[float(compl.real), float(compl.imag)] for compl in x])
y_real = np.array([[float(compl.real), float(compl.imag)] for compl in y])

M = 10
Batch_size = 1000

#X.shape == 153 x 1000 x 10 x 2 == 153 x Batch_size x M x 2
X =[[gen_bat_both(x_real, (5,5), j) for j in range(i,i+ Batch_size)] 
    for i in range(0, x_real.shape[0], Batch_size )]
X = X[:-1]
X = torch.FloatTensor(X).to(device)

Y = np.array([[y_real[j] for j in range(i,i + Batch_size)] 
    for i in range(0, y_real.shape[0] - Batch_size, Batch_size )]).reshape(X.shape[0],Batch_size,-1,2)
Y = torch.FloatTensor(Y).to(device)


Model = model_HC(M, Batch_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=1)

Model = train(Model,X,Y, criterion,optimizer, 300,1 )