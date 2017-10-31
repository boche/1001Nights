import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs, input_lens):
        # encoder_output: B x S x H
        # hidden: B x 1 x H
        batch_size, seq_len, hidden_size = encoder_outputs.size() 
        if self.method == 'general':
            # General score
            energy = self.attn(encoder_outputs).transpose(1, 2)
            attn_energies = hidden.bmm(energy)
        else:
            # Dot score
            attn_energies = hidden.bmm(encoder_outputs.transpose(1,2))
        for b in range(batch_size):
            if input_lens[b] < seq_len:
                attn_energies[b, 0, input_lens[b]:] = float("-inf")
        attn_energies = attn_energies.squeeze(1)
        return F.softmax(attn_energies).unsqueeze(1) # B x 1 x S
