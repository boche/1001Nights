import torch
import torch.nn as nn
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.vocab_size = args.vocab_size
        self.emb_size = args.emb_size
        self.word_layers = args.word_layers
        self.sent_layers = args.sent_layers
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.max_text_len = args.max_text_len
        self.rnn_model = args.rnn_model

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(self.emb, self.hidden_size, self.word_layers,
                self.sent_layers, self.dropout, self.rnn_model)
        self.decoder = DecoderRNN(self.emb, self.hidden_size, self.sent_layers,
                self.dropout, self.rnn_model)

    def forward(self, batch, is_volatile):
        (inputs, targets, inputs_eos_indices, targets_eos_indices,
                targets_kws, inputs_len, targets_len) = batch
        word_output, sent_output, sent_state = self.encoder(inputs, inputs_len,
                inputs_eos_indices, is_volatile)
        return self.decoder(targets, targets_kws, sent_state, targets_len)

    def sample(self):
        pass
