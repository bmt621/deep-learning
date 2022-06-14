from imports import *
from utils import *
from MHA_layer import *


class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropouts=0.2):
        super(EncoderLayer,self).__init__()
        
        self.norm1=ResidualLayerNorm(d_model)
        self.norm2=ResidualLayerNorm(d_model)
        self.mha=MultiheadAttention(d_model,num_heads)
        self.ff=PositionWiseFeedforward(d_model,d_ff)
        
    def forward(self,x,mask):
        mha,encoder_attn = self.mha(x,x,x,mask=mask)
        
        norm1=self.norm1(mha,x)
        
        ff=self.ff(norm1)
        
        norm2=self.norm2(ff,norm1)
        
        return norm2, encoder_attn
    
    
    
    