import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self, method, hidden_size, bidir):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.bidir = bidir
        
        if self.method == 'general':
            self.encoder_output_size = self.hidden_size * (2 if bidir else 1)
            self.attn = nn.Linear(self.encoder_output_size, hidden_size)

        if self.method == 'concat':
            self.encoder_output_size = self.hidden_size * (3 if bidir else 2)
            self.attn = nn.Linear(self.encoder_output_size, hidden_size)
            self.v = nn.Linear(self.hidden_size, 1)


    def forward(self, hidden, encoder_outputs, input_lens):
        # encoder_output: B x S x H
        # hidden: B x 1 x H
        batch_size, seq_len, hidden_size = encoder_outputs.size() 
        if self.method == 'general':
            # General score
            energy = self.attn(encoder_outputs).transpose(1, 2)
            attn_energies = hidden.bmm(energy)
        elif self.method == 'dot':
            # Dot score
            attn_energies = hidden.bmm(encoder_outputs.transpose(1,2))
        else:
            # Concat score
            hidden = hidden.repeat(1, seq_len, 1)  # B x S x H
            energy = F.tanh(self.attn(torch.cat([encoder_outputs, hidden], 2))) # B x S x H
            attn_energies = self.v(energy).transpose(1,2) # B x 1 x S

        for b in range(batch_size):
            if input_lens[b] < seq_len:
                attn_energies[b, 0, input_lens[b]:] = float("-inf")
        attn_energies = attn_energies.squeeze(1)
        return F.softmax(attn_energies).unsqueeze(1) # B x 1 x S
