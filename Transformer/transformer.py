from imports import *
from embeddings import Embeddings
from encoder import Encoder
from decoder import Decoder
from Generator import Generator

class Transformer(nn.Module):
    def __init__(self,src_vocab_len, trg_vocab_len,d_model,d_ff, num_layers, num_heads, src_pad_idx,\
         targ_pad_idx,dropout=0.2,device="cpu"):

        super(Transformer,self).__init__()

        self.num_heads = num_heads
        self.device = device
        self.encoder_embedding = Embeddings(src_vocab_len,d_model,src_pad_idx)
        self.decoder_embedding = Embeddings(trg_vocab_len,d_model,targ_pad_idx)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = targ_pad_idx

        self.encoder = Encoder(self.encoder_embedding,d_model,num_heads,
                               d_ff,num_layers,dropout,device)

        self.decoder = Decoder(self.decoder_embedding,d_model,num_heads,
                               d_ff,num_layers,dropout,device)
        
        self.generator = Generator(d_model,trg_vocab_len)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        
    def create_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).unsqueeze(2)
        return src_mask
    
    def create_trg_mask(self,trg):
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        mask = torch.ones((1,self.num_heads, trg.shape[1], trg.shape[1])).triu(1).to(self.device)

        mask = mask == 0

        trg_mask = trg_mask & mask

        return trg_mask

    def forward(self,src, trg):
        #shape(src): [B*src_seq_len]
        #shape(trg): [B*trg_seq_len]

        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)
        
        enc_outs, enc_mha_attn_weights = self.encoder(src,src_mask)
        dec_outs,_,enc_dec_attn_weights = self.decoder(trg,enc_outs, trg_mask,src_mask)
        
        #output = self.generator(dec_outs)

        return dec_outs,enc_dec_attn_weights




        

        

