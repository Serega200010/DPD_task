import torch
import torch.nn as nn
import torch.utils
import numpy as np
from collections import defaultdict,Counter
from torch.autograd import Variable
import torch.nn.functional as F



def to_tensor(A):
    a = [[[x for x in z] for z in b] for b in A]
    return torch.FloatTensor(a)


def NMSE(X, E):
    return 10*torch.log10((E.norm(dim=1)**2).sum()/(X.norm(dim=1)**2).sum())

def Svr_conv(vec, K):
    matr = []
    for i in range(0,K):
        matr.append([(lambda t: t ** i)(vec_i) for vec_i in vec])
    matr = np.array(matr)
    return matr

class AFIR(nn.Module):
    def __init__(self, M, D):
        super(AFIR, self).__init__()
        self.real = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).float()
        self.imag = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).float()
        self.real.weight.data.fill_(0.0)
        self.imag.weight.data.fill_(0.0)
        self.real.weight.data[0, 0, int((M-1)/2)+D] = 1.0
    def forward(self, x):
        '''
        r1 = self.real(x[:,0].view(1,1,-1))
        r2 = self.imag(x[:,1].view(1,1,-1))
        i1 = self.real(x[:,1].view(1,1,-1))
        i2 = self.imag(x[:,0].view(1,1,-1))'''
        r1 = self.real(x[:,0].reshape(1,1,-1))
        r2 = self.imag(x[:,1].reshape(1,1,-1))
        i1 = self.real(x[:,1].reshape(1,1,-1))
        i2 = self.imag(x[:,0].reshape(1,1,-1))       
        return torch.cat((r1-r2, i1+i2), dim=1)

# class Delay(AFIR):
#     def __init__(self,M,D):
#         super(Delay,self).__init__(M,D)
#         self.real.weight.requires_grad=False
#         self.imag.weight.requires_grad=False
#         self.imag.weight.data[0, 0, int((M-1)/2)+D] = 1.0
#     def forward(self, x):
#             r = self.real(x[:,0].view(1,1,-1))
#             i = self.imag(x[:,1].view(1,1,-1))
#             return torch.cat((r, i), dim=1)
class Delay(nn.Module):
    def __init__(self, M):
        super(Delay, self).__init__()
        self.op = nn.Sequential(
            nn.ConstantPad1d(M,0)
        )
    def forward(self, x):
        return self.op(x)[:,:,:x.shape[2]]
'''
class Prod_cmp(nn.Module):
    def __init__(self):
        super(Prod_cmp, self).__init__()
    def forward(self, inp1, inp2):
        r1 = inp1[:,0].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        r2 = inp1[:,1].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        i1 = inp1[:,1].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        i2 = inp1[:,0].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        return torch.cat((r1-r2, i1+i2), dim=1)'''
class Maslovsky_product(nn.Module):
    def __init__(self):
        super(Maslovsky_product, self).__init__()
    def forward(self, inp1, inp2):
        r1 = inp1[:,0].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        r2 = inp1[:,1].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        i1 = inp1[:,1].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        i2 = inp1[:,0].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        return torch.cat((r1-r2, i1+i2), dim=1)



def prod_cmp( inp1, inp2):
        r1 = inp1[:,0].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        r2 = inp1[:,1].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        i1 = inp1[:,1].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        i2 = inp1[:,0].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        return torch.cat((r1-r2, i1+i2), dim=1)


class Prod_cmp(nn.Module):
    def __init__(self):
        super(Prod_cmp, self).__init__()
    def forward(self, I1, I2):
        ans = []
        for i in range(I1.shape[0]):
            ans.append(list(prod_cmp(I1[i], I2[i]).view(-1,2)))
        ans = to_tensor(ans)

        return ans

class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()
    def forward(self, inp1, inp2):
        return inp1.cuda() + inp2



class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, x):
        out = x.norm(dim=2, keepdim=True)
        return out

class Polynomial(nn.Module):
    def __init__(self, Poly_order,passthrough=False):
        super(Polynomial, self).__init__()
        self.order = Poly_order
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights = nn.Parameter(torch.zeros((2, Poly_order), device=device, dtype=torch.float64), requires_grad=True)
        if passthrough:
            self.weights.data[0, 1] = 1
#         else:
#             torch.linspace(0,1,Poly_order,out=self.weights[0,:],device=device,requires_grad=True)
        self.Abs = ABS()
    def forward(self, x):
        #print(x.shape)
        out = torch.zeros_like(x)
        x = self.Abs(x).view(1, -1)
        #print(x.shape)
        for i in range(self.order):
            out[:, 0] += self.weights[0, i]*torch.pow(x,i)
            out[:, 1] += self.weights[1, i]*torch.pow(x,i)
        return out


def update_history(hist,iter_num, val_acc, val_loss, time):
    hist['iter'].append(iter_num)
    hist['time'].append(time)
    hist['train_loss'].append(val_loss.item())
        # self.hist['norm_coeffs'].append(train_loss.item())
    hist['train_loss_db'].append(val_acc.item())


def gen_bat(x, M, i):
    if i+M > len(x):
        return (np.vstack((x[i:len(x)],np.zeros((M+i-len(x),2))))).tolist()
    if i < M:
        x_batch = np.vstack((np.zeros((M-i,2)),x[0:i]))
        return x_batch.tolist()
    if i >= M:
        x_batch =  x[i:M+i]
        return x_batch.tolist()

def gen_bat_back(x, M, i):
   # if i+M > len(x):
    #    return (np.vstack((x[i:len(x)],np.zeros((M+i-len(x),2))))).tolist()
    if i < M:
        x_batch = np.vstack((np.zeros((M-i,2)),x[0:i]))
        return x_batch.tolist()
    if i >= M:
        x_batch =  x[i-M:i]
        return x_batch.tolist()


def gen_bat_both(x, params, i):
    M1 = params[0]
    M2 = params[1]

    if i - M1 < 0:
        x_batch_back = np.vstack((np.zeros((M1 - i,2)), x[0:i]))
    else:
        x_batch_back = x[i - M1 :i]
    if i + M2 > len(x):
        x_batch_forward = (np.vstack((x[i:len(x)],np.zeros((M2+i-len(x),2)))))
    else:
        x_batch_forward = x[i:i+M2]
    
    x_batch = np.vstack((x_batch_back,x_batch_forward))
    return x_batch


    return ans

#def form_the_batches(X, batch_size = 1):  #x1 ... xn -> x1...x

    '''
def gen_bat1(x, M, i):
    if i+M > len(x):
        return (np.hstack((x[i:len(x)],np.zeros((M+i-len(x)))))).tolist()
    if i < M:
        x_batch = np.hstack((np.zeros((M-i)),x[0:i]))
        return x_batch.tolist()
    if i >= M:
        x_batch =  x[i:M+i]
        return x_batch.tolist()
import torch
'''

def eval_model(valid_queue, model,criterion):
    for step, (valid) in enumerate(valid_queue):
        model.eval()
        input_batch = Variable(valid[:,:,:1],requires_grad=False).permute(2,1,0).cuda()
        desired = Variable(valid[:,:,1:],requires_grad=False).permute(2,1,0).cuda()
        out = model.forward(input_batch)


        loss=criterion(out,desired)
        #draw_spectrum(input_batch,desired,out)

        
        accuracy = NMSE(input_batch, out-desired)
    return loss,accuracy

def train_of_epoch(train_queue, model, criterion, optimizer):
    for step, (train) in enumerate(train_queue):

        input_batch = Variable(train[:,:,:1],requires_grad=False).permute(2,1,0).cuda()
        desired = Variable(train[:,:,1:],requires_grad=False).permute(2,1,0).cuda()
        optimizer.zero_grad()
        out = model.forward(input_batch)
        loss = criterion(out, desired)

        loss.backward()

        optimizer.step()



def train(train_queue, valid_queue, model, criterion, optimizer,n_epoch,
          log_every=1):
    min_loss=0
    for it in range(n_epoch):
        model.train(True)
        train_of_epoch(train_queue, model, criterion, optimizer)
        if it%log_every==0:
            loss_v,accuracy_v=eval_model(valid_queue,model, loss_fn)
            print('Loss = ',loss_v.cpu().detach().numpy(), 'Accuracy = ', accuracy_v.cpu().detach().numpy(), 'dbs')












            '''
            if save_flag:
                with open(path_to_experiment + '/hist.pkl', 'wb') as output:
                    pickle.dump(hist, output)

                    torch.save(model.state_dict(), path_to_experiment + '/model.pt')
                if hist['train_loss_db'][-1] < min_loss:
                            min_loss = hist['train_loss_db'][-1]
                            torch.save(model.state_dict(), path_to_experiment + '/best_model.pth')'''