import numpy as np
import torch
from torch import nn
from torch.nn import functional as f
from torch.autograd import Variable
from tensorflow import keras 
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist 

(x_train,y_train),(x_test,y_test)=mnist.load_data() # load mnist datasets
x_train=keras.utils.normalize(x_train,axis=1)  # normalise the datasets to grascale
x_test=keras.utils.normalize(x_test,axis=1)    #.





# COMPUTE OUTPUT SHAPE FOR EACH CONVOLUTION OPERATION

def conv_shape(input_shape,channels,kernel=(3,3),max_kernel=(2,2),stride=(1,1),max_stride=(2,2),pad=(0,0),max_pad=(0,0),dilation=(1,1),max_dilation=(1,1),max_pooling=False):
    
    if(len(input_shape) != 2):
        print('expected 2D shape for a convolution')
        return
    
    n_block=len(channels)
    new_output=input_shape
    
    if(max_pooling):
        
        
        for i in range(n_block):
        
            output=pool(new_output,kernel,stride,pad,dilation)
            maxpool=pool(output,max_kernel,max_stride,max_pad,max_dilation)
            new_output=maxpool
            
        
    else:
        
        for i in range(n_block):
            output=pool(new_output,kernel,stride,pad,dilation)
            new_output=output
            
    
    if(new_output[0]<=0 and new_output[1]<=0):
        print('cannot compute for 0 convolutions')
        return
    
    
    return(new_output)
    


def pool(h_w,kernel=(2,2),stride=(2,2),pad=(0,0),dilation=(1,1)):
     
    h=np.floor(((h_w[0] + (2*pad[0]) - (dilation[0]*(kernel[0] - 1) ) - 1)/ stride[0]) + 1)
    w=np.floor(((h_w[1] + (2*pad[1]) - (dilation[1]*(kernel[1] - 1) ) - 1)/ stride[1]) + 1)
    
    return((int(h),int(w)))

# CONVOLUTION CLASS FOR MNIST DATASETS

#LETS COMPUTE OUTPUT SHAPE USING THE CONV_SHAPE FUNCTION ABOVE
conv_shape1=conv_shape((28,28),channels=[10],kernel=(5,5),max_kernel=(2,2),max_pooling=True)
conv_shape2=conv_shape(conv_shape1,channels=[20],kernel=(5,5),max_kernel=(2,2),max_pooling=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn1=nn.Conv2d(1,10,kernel_size=(5,5)) # arg: input features,output features
        self.cnn2=nn.Conv2d(10,20,kernel_size=(5,5))
        self.mp=nn.MaxPool2d((2,2))
        
        self.ffn1=nn.Linear(conv_shape2[0]*conv_shape2[0]*20,100)
        self.ffn2=nn.Linear(100,10)

    def forward(self,x):
        output=f.relu(self.mp(self.cnn1(x)))
        output=f.relu(self.mp(self.cnn2(output)))

        in_size=x.size(0)
        output=output.view(in_size,-1)
        
        output=f.relu(self.ffn1(output))
        output=f.relu(self.ffn2(output))
        output=f.log_softmax(output,dim=1)

        return(output)


# A DATA PROCESSOR CLASS TO FEED TO THE DATA_LOADER
class dataprocess():
    
    def __init__(self,X,Y):
        
        self.len=X.shape[0]
        
        self.x_data=torch.FloatTensor(X.reshape(self.len,1,28,28))
        self.y_data=torch.FloatTensor(Y)
        
        
    
    def __getitem__(self,index):
        return(self.x_data[index],self.y_data[index])
    
    def __len__(self):
        return(self.len)
    

    
train_data=dataprocess(x_train,y_train)
test_data=dataprocess(x_test,y_test)

DataLoader=torch.utils.data.DataLoader

train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=32,shuffle=True)

cnn=CNN()

# LETS DEFINE THE LOSS AND THE OPTIMIZER
loss_fn=nn.CrossEntropyLoss()
optim=torch.optim.Adam(cnn.parameters(),lr=0.01)

# LETS TRAIN OUR CNN
epoch=300
cnn.train()
for i in range(epoch):
    for batch_indx,(data_x,data_y) in enumerate(train_loader):
        train_x,train_y=Variable(data_x),Variable(data_y).type(torch.LongTensor)
        pred=cnn(train_x)
        loss=loss_fn(pred,train_y)
        loss.backward()
        optim.step()
        optim.zero_grad()
    if i % 100==0:
        print('cost: ',loss.item())
    


# LETS TEST OUR MODEL

cnn.eval()
test_loss=0.0
correct=0.0
for data,target in test_loader:
    data_x,data_y=Variable(data),Variable(target)
    output=cnn(data_x)
    data_y=data_y.type(torch.LongTensor)
    test_loss+=loss_fn(output,data_y).item()
    pred=output.data.max(1,keepdim=True)[1]
    correct+=pred.eq(data_y.data.view_as(pred)).cpu().sum()
test_loss/=len(test_loader.dataset)
print('\ntest: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),
                                                                            100.*correct/len(test_loader.dataset)))


