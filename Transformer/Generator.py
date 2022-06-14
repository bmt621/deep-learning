from imports import nn,f

class Generator(nn.Module):

    def __init__(self,d_model,trg_vocab_len):
        super(Generator,self).__init__()
        self.proj=nn.Linear(d_model,trg_vocab_len)


    def forward(self,x):
        return f.log_softmax(self.proj(x),dim=-1)