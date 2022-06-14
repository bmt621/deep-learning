from imports import *
from utils import PositionalEncoding
from embeddings import *
from Encoder_layer import *



class Encoder(nn.Module):
    def __init__(self,embeddings:Embeddings,d_model,num_heads,d_ff,num_layers,dropouts=0.2,device='cpu'):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.embeddings = embeddings
        
        self.PE = PositionalEncoding(d_model,device=device)
        encoder_layer = EncoderLayer(d_model,num_heads=num_heads,d_ff=d_ff,dropouts=dropouts)
        
        self.encoders = clones(encoder_layer,num_layers)
        
    
    def forward(self,x,mask):
        
        embedding = self.embeddings(x)
        encodings = self.PE(embedding)
        
        for encoder in self.encoders:
            encoding, attn_weights = encoder(encodings, mask)
            
        
        return encoding, attn_weights
    
    