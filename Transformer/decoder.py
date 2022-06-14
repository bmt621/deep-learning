from imports import *
from embeddings import Embeddings
from utils import PositionalEncoding,clones
from decoder_layer import DecoderLayer
class Decoder(nn.Module):
    def __init__(self,embeddings:Embeddings,d_model,num_heads,d_ff,num_layers,dropouts=0.2,device='cpu'):
        super(Decoder,self).__init__()

        self.embeddings = embeddings

        self.pos_encode= PositionalEncoding(d_model,device = device)
        self.dropout = nn.Dropout(dropouts)
        
        self.decoders = clones(DecoderLayer(d_model,num_heads,d_ff=d_ff,dropout = dropouts),
                               num_layers)



    def forward(self,trg,enc_output,trg_mask,src_mask):

        embedding = self.embeddings(trg)
        encoding = self.pos_encode(embedding)

        for decoder in self.decoders:
            encoding, masked_attn_weights, enc_dec_attn_weights = decoder(encoding,enc_output,
            trg_mask,src_mask)

        
        return (encoding, masked_attn_weights,enc_dec_attn_weights)

