import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.vocab_size = args.vocab_size
        self.emb_size = args.emb_size
        self.nlayers = args.nlayers
        self.hidden_size = args.hidden_size
        self.teach_ratio = args.teach_ratio
        self.dropout = args.dropout
        self.max_title_len = args.max_title_len
        self.attn_model = args.attn_model
        self.rnn_model = args.rnn_model
        self.bidir = args.bidir
        if self.bidir:
            self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.dropout, self.rnn_model, self.bidir)
        self.decoder = DecoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.teach_ratio, self.dropout, self.rnn_model, self.bidir, self.attn_model)

    def forward(self, inputs, input_lens, targets):
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        if self.bidir:
            encoder_hidden = encoder_hidden.view(self.nlayers, -1, self.hidden_size * 2)
            encoder_hidden = F.tanh(self.linear(encoder_hidden))
        logp = self.decoder(targets, encoder_hidden, encoder_output, input_lens)
        return logp

    def summarize(self, inputs, input_lens, beam_search=True):
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        if self.bidir:
            encoder_hidden = encoder_hidden.view(self.nlayers, -1, self.hidden_size * 2)
            encoder_hidden = F.tanh(self.linear(encoder_hidden))
        logp, symbols = None, None
        if beam_search:
            logp, symbols, attns = self.decoder.summarize_bs(encoder_hidden,
                    self.max_title_len, encoder_output, input_lens)
        else:
            logp, symbols, attns = self.decoder.summarize(encoder_hidden,
                    self.max_title_len, encoder_output, input_lens)
        return logp, symbols, attns
