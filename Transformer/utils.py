from imports import *

def clones(module,N):
    return (nn.ModuleList([module for _ in range(N)]))


def create_mask(size):
    return(torch.ones((1,size,size)).triu(1) == 0)


class ResidualLayerNorm(nn.Module):
    def __init__(self,d_model,dropout=0.2):
        super().__init__()
        self.layer_norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,residual):
        
        out=self.layer_norm(self.dropout(x)+residual)
        
        return out
    

#              Position Wise FeedForward
class PositionWiseFeedforward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionWiseFeedforward,self).__init__()
        
        self.ff1=nn.Linear(d_model,d_ff)
        self.ff2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        return(self.ff2(self.dropout(f.relu(self.ff1(x)))))
    

#              Positional Encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len=300,dropouts=0.2,device='cpu'):
        super(PositionalEncoding,self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropouts)
        
        pe = torch.zeros(max_seq_len,self.d_model).to(device)
        pos = torch.arange(0,max_seq_len).unsqueeze(1).float()
        
        two_i = torch.arange(0,d_model,step=2).float()
        div_term = torch.pow(10000,(two_i/torch.Tensor([d_model]))).float()
        
        pe[:,0::2] = torch.sin(pos/div_term)
        pe[:,1::2] = torch.cos(pos/div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)
        
    def forward(self,x):
        pe = self.pe[:,:x.shape[1]].detach()
        
        x = x.add(pe)
        
        return self.dropout(x)
     