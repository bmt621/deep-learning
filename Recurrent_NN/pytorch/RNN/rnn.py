import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as f
from torch.autograd import Variable
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale


#lets load the boston datasets

boston=load_boston()
feat=scale(np.array(boston.data))
label=np.array(boston.target)


train_data_x=feat[:len(feat)-100]
train_data_y=label[:len(label)-100]

#validate on last 100 datapoints
val_data_x=feat[len(feat)-100:]
val_data_y=label[len(label)-100:]


#this function create the input sequence to the rnn network of 50 timestep
def process(X,Y):
    x=[]
    y=[]
    get_y=Y
    get_x=X
    for i in range(50,len(get_x)):
        x.append(get_x[i-50:i])
        y.append(get_y[i])
        
    return np.array(x),np.array(y)


new_x,new_y=process(train_data_x,train_data_y)
new_val_x,new_val_y=process(val_data_x,val_data_y)


x_train,y_train=torch.FloatTensor(new_x.reshape(356,50,13)),torch.FloatTensor(new_y.reshape(-1,1))
x_val,y_val=torch.FloatTensor(new_val_x.reshape(50,50,13)),torch.FloatTensor(new_val_y.reshape(-1,1))


class RNN(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,n_layers):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        
        self.rnn=nn.RNN(input_size,hidden_size,num_layers=n_layers,batch_first=True)
        self.i2o=nn.Linear(hidden_size,output_size)
        
        self.n_layers=n_layers
        
    def forward(self,input):
        batch_size=input.size(0)
        hidden=self.initHidden(batch_size)
        r_out,hidden=self.rnn(input,hidden)
        out=self.i2o(hidden[-1])
        
        return(out,hidden)
    
    def initHidden(self,batch_size):
        return torch.zeros(self.n_layers,batch_size,self.hidden_size)
    

n_hidden=10
n_input=13
output_size=1
rnn=RNN(n_input,output_size,hidden_size=n_hidden,n_layers=2)


ls_fn1=nn.MSELoss()
lr=0.01
optim=torch.optim.Adam(rnn.parameters(),lr=lr)


iters=1000
rnn.train()
train_x,train_y=Variable(x_train),Variable(y_train)
for i in range(iters):
    out,hidden=rnn(train_x)
        
    loss=ls_fn1(out,train_y)
    loss.backward()
    optim.step()
    optim.zero_grad()
    
    
    if(i % 100 == 0):
        print('cost: ',loss.item())



