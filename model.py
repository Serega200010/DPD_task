import numpy as np
import torch
import torch.nn as nn
from blocks import *


def NMSE(X, E):
    return 10*torch.log10((E.norm(dim=1)**2).sum()/(X.norm(dim=1)**2).sum())

class sigm_layer(nn.Module):
    def __init__(self):
        super(sigm_layer,self).__init__()
        self.activation = nn.Sigmoid()
    def forward(self, X: torch.tensor) -> torch.tensor:
        return self.activation(X)

class mult_layer(nn.Module): #Сделать это комплексным!!!!
    def __init__(self):
        super(mult_layer,self).__init__()
    def forward(self,X: torch.tensor,Y: torch.tensor) -> torch.tensor:
        try: 
            ans = X*Y
        except RuntimeError:
            print('Mult_layer_error: wrong dimensions: X : {}, Y : {}'.format(X.shape, Y.shape))
            exit('Mult_layer_error: wrong dimensions')
        return ans

class sum_layer(nn.Module):
    def __init__(self):
        super(sum_layer,self).__init__()
    def forward(self,X: torch.tensor,Y: torch.tensor) -> torch.tensor:
        try: 
            ans = X+Y
        except RuntimeError:
            exit('Sum_layer_error: wrong dimensions')
        return ans

class model(nn.Module):
    def __init__(self, input_size: int):
        super(model,self).__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList(
                                    [ABS(), sigm_layer(), nn.Linear(input_size, input_size), 
                                    mult_layer(), sum_layer(), nn.Conv2d(in_channels = input_size, out_channels = 1, kernel_size = 1)])#Свертка...
    def forward(self, X: torch.tensor):
        tmp_condition = X
        tmp_condition = self.layers[0](tmp_condition)
        tmp_condition = self.layers[1](tmp_condition)
        tmp_condition = self.layers[2](tmp_condition.view(-1,1,self.input_size)).view(-1,self.input_size,1)#.transpose(-1,0))
        tmp_condition = self.layers[3](X, tmp_condition)#.transpose(-1,0))
        tmp_condition = self.layers[4](X, tmp_condition)
        ans = self.layers[5](tmp_condition.view(-1,self.input_size,1,1))
        return ans

def train(model, x,y,criterion, optimizer,num_epoch, show_freq = 10): #Улучшить для батчей!!
       for t in range(num_epoch): 
        i = 0
        print(t%show_freq)
        for X in x:
            y_pred = model(X)
            loss = criterion(y_pred.view(y[i].shape[0],2,-1,1), y[i].view(-1,2,1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1
        if t % show_freq == 0:
         print(t, loss.item())
       return model

def form_the_batches(X, batch_size: torch.tensor) -> torch.tensor:
    return torch.split(X, batch_size, dim = 0)


def form_the_u(X):
   U = torch.FloatTensor([ [[x[i].item()**(2*j) for i in range(x.shape[0])] for j in range(x.shape[0]) ]   for x in X] )
   return U

def to_tensor(A):
    a = [[x for x in z] for z in A]
    return torch.FloatTensor(a)


class Hard_Conv(nn.Module):
    def __init__(self,M = 10, parameters = [[5,7,5] , [7,10,8] , [10,7,5] , [7,1,10], [7,1,10]]):
        super(Hard_Conv,self).__init__()
        self.bn1 = nn.BatchNorm2d(M)
        self.M = M
        self.x1 = nn.Conv1d(5, 7, 10).cuda()
        self.x12 = nn.Conv1d(7,10,1).cuda()
        self.x21 = nn.Conv1d(10,7,1).cuda()
        self.x2 = nn.Conv1d(7,M,1).cuda()
        self.x2_1 = nn.Conv1d(7,M,1).cuda()
    def forward(self, X, K = 5): # X.shape == Batch_size x M x 1
        output = []
        m = nn.Sigmoid()
        i = 0
        for x in X: #!!!!!
                i+=1
                #print('it is a Brand new element ', i)
                a = to_tensor(Svr_conv(x,K))
                U = torch.FloatTensor(a).view(-1,self.M,K,1).cuda()
                #tmp_condition = self.bn1(U).view(self.M, 1, -1)
                tmp_condition = self.bn1(U).view(1,-1,self.M)
                tmp_condition = self.x1(tmp_condition)
                tmp_condition = m(tmp_condition)
                tmp_condition = self.x12(tmp_condition)
                tmp_condition = m(tmp_condition)
                tmp_condition = self.x21(tmp_condition)
                tmp_condition = m(tmp_condition)

                real = m(self.x2(tmp_condition))
                comp= m(self.x2_1(tmp_condition))
                output.append((torch.cat((real, comp), 0)).view(self.M,2))
                #output.append([real, comp])
        #output = torch.FloatTensor(output)
        output = np.array(output)
        return output



class conv_matrix_layer(nn.Module):
    def __init__(self, M: int, batch_size: int):
        super(conv_matrix_layer,self).__init__()
        self.U = torch.zeros((batch_size,M,M ))
        self.Size = M
        self.bn1 = nn.BatchNorm1d(M)
        self.Conv = nn.Conv2d(in_channels = M**2, out_channels = M, kernel_size = 1)
    def forward(self, X: torch.tensor):
        #print(X.shape)
        #self.U[:,:,:] = torch.FloatTensor([[[float(x[i].item())**(2*j) for i in range(self.Size)] for j in range(self.Size)] for x in X])
        self.U = self.bn1(form_the_u(X))

        ans = [list(self.Conv(self.U[i].view(1,self.Size**2,1,-1)).view(self.Size))  for i in range(self.U.shape[0])]
        #print(ans[0].shape)
        ans = torch.FloatTensor(ans)
        #return torch.FloatTensor([self.Conv(self.U[i,:,:].view(1,self.Size**2,1,-1)) for i in range(self.U.shape[0])])
        return ans





class model_1(nn.Module):
    def __init__(self, input_size: int, batch_size: int):
        super(model_1,self).__init__()
        self.input_size = input_size
        self.BS = batch_size
        self.bn1 = nn.BatchNorm1d(input_size)
        self.layers = nn.ModuleList(
                                    [ABS(), sigm_layer(), nn.Linear(input_size, input_size), conv_matrix_layer(input_size, batch_size),
                                    mult_layer(), sum_layer(), nn.Conv2d(in_channels = input_size, out_channels = 1, kernel_size = 1)])#Свертка...
    def forward(self, X: torch.tensor):
        #tmp_condition = self.bn1(X)
        tmp_condition = X
        tmp_condition = self.layers[0](tmp_condition)
        tmp_condition = self.layers[1](tmp_condition)
        tmp_condition = self.layers[2](tmp_condition.view(-1,1,self.input_size)).view(self.input_size,-1)#.transpose(-1,0))
        tmp_condition = self.layers[3](tmp_condition.view(self.BS, -1)).view(-1,self.input_size,1)
        tmp_condition = self.layers[4](X, tmp_condition)#.transpose(-1,0))
        tmp_condition = self.layers[5](X, tmp_condition)
        ans = self.layers[6](tmp_condition.view(-1,self.input_size,1,1))
        return ans

def train(model, x,y,criterion, optimizer,num_epoch, show_freq = 10): #Улучшить для батчей!!
       for t in range(num_epoch): 
        i = 0
        for X in x:
            if i%10 ==0:
                print('Element ', i)
            y_pred = model(X)
            loss = criterion(y_pred.view(y[i].shape[0],2,-1,1), y[i].view(-1,2,1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i+=1
        if t % show_freq == 0:
         print(t, loss.item())
       return model

#M === 10!!!

'''nn.Conv1d(2,input_size,1)'''

class model_HC(nn.Module):
    def __init__(self, input_size: int, batch_size: int):
        super(model_HC,self).__init__()
        self.input_size = input_size
        self.BS = batch_size
        self.layers = nn.ModuleList([nn.Linear(input_size, input_size),Hard_Conv(), nn.Conv1d(2,1,1), nn.Linear(input_size, 2) ]).cuda()
        self.Abs = ABS()
        self.sigm = nn.Sigmoid()
        self.multiplication = Prod_cmp()
        self.sum = Sum()
    def forward(self, X): #X: BS x M x 2
        tmp_condition = self.Abs(X).view(-1, self.BS, self.input_size).cuda()
        tmp_condition = self.layers[0](tmp_condition).view(self.BS, self.input_size, -1)
        tmp_condition = self.sigm(tmp_condition)
        tmp_condition = self.layers[1](tmp_condition)
        tmp_condition = self.multiplication(tmp_condition, X)#.view(self.BS, 2, -1)
        tmp_condition = self.sum(tmp_condition,X).view(self.BS, 2, -1)
        tmp_condition = self.layers[2](tmp_condition)
        tmp_condition = self.sigm(tmp_condition)
        ans = self.layers[3](tmp_condition)
        return ans.view(self.BS, -1)





















        #self.x2 = nn.Conv2d(in_channels = parameters[3][0], out_channels = parameters[3][1], kernel_size = parameters[3][2])
        #self.x2_1 = nn.Conv2d(in_channels = parameters[4][0], out_channels = parameters[4][1], kernel_size = parameters[4][2])



        #self.x21 = nn.Conv2d(in_channels = parameters[2][0], out_channels = parameters[2][1], kernel_size = parameters[2][2])
                #self.x12 = nn.Conv2d(in_channels = parameters[1][0], out_channels = parameters[1][1], kernel_size = parameters[1][2])
                 #       self.x1 = nn.Conv2d(in_channels = parameters[0][0], out_channels = parameters[0][1], kernel_size = parameters[0][2])





'''
class layer(nn.Module):
    def __init__(self, num_cells = 1, cell_type = 'AFIR', *params):
        super(Net,self).__init__()
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.params = params
    def activate():
        if cell_type == 'AFIR':
            M, D = self.params[0], self.params[1]
            m = AFIR(M,D)
        if cell_type == 'Delay':
            m = 
        self.cells = nn.ModuleList()

class Net(nn.Module):
    def __init__(self, layers):
        super(Net,self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
'''
