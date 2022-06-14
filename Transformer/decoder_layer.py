from ast import Mult
from MHA_layer import MultiheadAttention
from imports import *
from utils import *

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.2):
        super(DecoderLayer,self).__init__()

        self.norm1 = ResidualLayerNorm(d_model)
        self.norm2 = ResidualLayerNorm(d_model)
        self.norm3 = ResidualLayerNorm(d_model)

        self.masked_mha = MultiheadAttention(d_model,num_heads)
        self.enc_dec_mha = MultiheadAttention(d_model,num_heads)

        self.ff = PositionWiseFeedforward(d_model,d_ff)


    def forward(self,trg,enc_out, trg_mask, src_mask):

        masked_mha,masked_attn_weights = self.masked_mha(trg,trg,trg,mask = trg_mask)

        norm1 = self.norm1(masked_mha,trg)

        enc_dec_mha ,enc_dec_attn_weights = self.enc_dec_mha(
            norm1,enc_out,enc_out,mask=src_mask)

        norm2 = self.norm2(enc_dec_mha, norm1)

        ff = self.ff(norm2)

        norm3 = self.norm3(ff,norm2)

        return norm3, masked_attn_weights, enc_dec_attn_weights



