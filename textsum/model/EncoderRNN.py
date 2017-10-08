import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nlayers):
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size = hidden_size,
                num_layers = nlayers)

    def forward(self, xin, input_lengths):
        """
        xin: batch, seq_len
        """
        xin = Variable(xin)
        xin_emb = self.emb(xin)
        xin_pack = nn.utils.rnn.pack_padded_sequence(xin_emb, input_lengths,
                batch_first=True)
        output_pack, hidden = self.rnn(xin_pack)
        output, output_len = nn.utils.rnn.pad_packed_sequence(output_pack,
                batch_first=True)
        return output, hidden
