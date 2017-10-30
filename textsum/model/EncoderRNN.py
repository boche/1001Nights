import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb, hidden_size, nlayers, dropout):
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.emb = emb
        # self.dropout = nn.Dropout(dropout)
        emb_size = self.emb.weight.size(1)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size = hidden_size,
                dropout = dropout, num_layers = nlayers, batch_first = True)

    def forward(self, xin, input_lengths):
        """
        xin: batch, seq_len
        """
        # xin_emb = self.dropout(self.emb(Variable(xin)))
        xin_emb = self.emb(Variable(xin))
        xin_pack = nn.utils.rnn.pack_padded_sequence(xin_emb, input_lengths,
                batch_first=True)
        output_pack, hidden = self.rnn(xin_pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_pack,
                batch_first=True)
        return output, hidden
