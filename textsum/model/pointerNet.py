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
        generator_input_size = emb_size + 2 * hidden_size
        self.generator = nn.Linear(generator_input_size, 1, bias=True)    

    def forward(self, context, rnn_output, input_emb):
        context = context.squeeze(1)         # B x H
        rnn_output = rnn_output.squeeze(1)   # B x H

        # # debugging info
        # print('context: ', type(context), context.size())
        # print('rnn_output: ', type(rnn_output), rnn_output.size())
        # print('input_emb: ', type(input_emb), input_emb.size())

        gen_input = torch.cat((input_emb, context, rnn_output), 1)
        prob_gen = F.sigmoid(self.generator(gen_input))
        # print('prob_gen: ', type(prob_gen), prob_gen.size())
        return prob_gen
