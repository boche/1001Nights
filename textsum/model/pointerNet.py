import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PointerNet(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(PointerNet, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        # Linear Layers for Prob_generator calculation 
        self.gen_inp = nn.Linear(self.emb_size, 1)      # partial score from decoder input
        self.gen_ctx = nn.Linear(self.hidden_size, 1)   # partial score from context vector
        self.gen_dec = nn.Linear(self.hidden_size, 1)   # partial score from decoder hidden state 
        self.generator = nn.Linear(1, 1, bias=True)     # aggregated scalar 

    def forward(self, context, rnn_output, input_emb):
        context = context.squeeze()         # B x H
        rnn_output = rnn_output.squeeze()   # B x H
        
        # debugging info
        print('context size: ', context.size())
        print('rnn_output size: ', rnn_output.size())
        print('input_emb size: ', input_emb.size())
        
        score = self.gen_inp(input_emb) + self.gen_ctx(context) + self.gen_dec(rnn_output)
        prob_gen = F.sigmoid(self.generator(score))
        print(score.size(), prob_gen.size())
        return prob_gen
