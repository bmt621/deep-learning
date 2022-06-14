from imports import *
from embeddings import *
from encoder import *
from transformer import Transformer



'''
toy_tokenizers = torch.LongTensor([[1,2,3,4,0,0]])
toy_embedding = Embeddings(5,d_model=4,padding=0)
toy_vocab=toy_embedding(toy_tokenizers)

toy_encoder = Encoder(toy_embedding,d_model = 4,num_heads=2,d_ff = \
    10,num_layers=8)


token_output,token_attn_weights=toy_encoder(toy_tokenizers)

print(token_output)

'''

inp_token = torch.LongTensor([[1,2,3,4,0,0]])
out_token = torch.LongTensor([[1,3,3,2,0,0]])

transformer = Transformer(5,5,4,2,4,2,0,0)

out,enc_dec_attn_weight=transformer(inp_token,out_token)

print(enc_dec_attn_weight)