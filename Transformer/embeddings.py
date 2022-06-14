from imports import *


class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model,padding):
        super(Embeddings,self).__init__()
        
        self.d_model = d_model
        self.embed=nn.Embedding(vocab_size,d_model,padding_idx=padding)
        
    def forward(self,x):
        embedding = self.embed(x)
        
        return embedding * math.sqrt(self.d_model)
    
    