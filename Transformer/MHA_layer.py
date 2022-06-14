from imports import *
from utils import *




class MultiheadAttention(nn.Module):
     
    def __init__(self,d_model,num_heads,dropouts=0.1):
        super(MultiheadAttention,self).__init__()
        self.d=d_model//num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model,d_model)
        self.linear_k = nn.Linear(d_model,d_model)
        self.linear_v = nn.Linear(d_model,d_model)
        
        self.mha_linear=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropouts)
    
    def scaled_dot_product(self,Q,K,V,mask=None):
        
        assert Q.size(-1) == K.size(-1) == V.size(-1), f"dimentions of each matrix should be the same"
        
        

        qk=torch.matmul(Q,K.permute(0,1,3,2))/math.sqrt(self.d) # (b,seq_len,seq_len)
        
        if mask is not None:
            qk=qk.masked_fill(mask == 0,-1e9)
            
            
        attn_weights=f.softmax(qk,dim=-1)                     # (b,seq_len,seq_len)
        
        output=torch.matmul(attn_weights,V)                   #(b,seq_len,self.d)
        
        return output,attn_weights
    
    def forward(self,prev_q,prev_k,prev_v,mask=None):
        #shape(x) [b,seq_len,d_model]
        
        Q=self.linear_q(prev_q)               #(b,seq_len,self.d)*d_model
        K=self.linear_k(prev_k)               #(b,seq_len,self.d)*d_model
        V=self.linear_v(prev_v)               #(b,seq_len,self.d)*d_model
        
        batch_size = prev_q.size(0)

        Q = Q.reshape(batch_size,self.num_heads,-1,self.d)
        K = K.reshape(batch_size,self.num_heads, -1,self.d)
        V = V.reshape(batch_size, self.num_heads, -1,self.d)


        output, attn_weights = self.scaled_dot_product(Q,K,V)
        output = output.reshape(batch_size,-1,self.d_model)

        project=self.dropout(self.mha_linear(output))
        
        return project,attn_weights
    
    
            