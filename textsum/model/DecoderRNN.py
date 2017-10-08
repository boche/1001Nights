import torch
import torch.nn as nn
from torch.autograd import Variable

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, nlayers, encoder_emb):
        """
        **encoder_emb** is the embedding layer from the encoder, decoder will use
        the same embedding layer to reduce parameters.
        """

        super(DecoderRNN, self).__init__()
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size = hidden_size,
                num_layers = nlayers)
        self.output_size = vocab_size
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.emb = encoder_emb

    def forward(self, target):
        """
        target: batch_size, seq_len
        """
        batch_size = target.size(0)
        teacher_forcing_ratio = 0.5
