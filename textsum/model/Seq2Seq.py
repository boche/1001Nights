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
            if self.rnn_model == 'lstm':
                self.linear_cell = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.dropout, self.rnn_model, self.bidir)
        self.decoder = DecoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.teach_ratio, self.dropout, self.rnn_model, self.bidir, self.attn_model)

    def bidirTrans(self, encoder_hidden, isCell=False):
        encoder_hidden = torch.cat([encoder_hidden[0::2, :, :], encoder_hidden[1::2, :, :]], 2)
        if isCell:
            encoder_hidden = F.tanh(self.linear_cell(encoder_hidden))
        else:
            encoder_hidden = F.tanh(self.linear(encoder_hidden))
        return encoder_hidden

    def forward(self, inputs, input_lens, targets):
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        if self.bidir:
            if self.rnn_model == 'gru':
                encoder_hidden = self.bidirTrans(encoder_hidden)
            else:
                hidden_state = self.bidirTrans(encoder_hidden[0])
                cell_state = self.bidirTrans(encoder_hidden[1], True)
                encoder_hidden = (hidden_state, cell_state)
        logp = self.decoder(targets, encoder_hidden, encoder_output, input_lens)
        return logp

    def summarize(self, inputs, input_lens, beam_search=True):
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        if self.bidir:
            if self.rnn_model == 'gru':
                encoder_hidden = self.bidirTrans(encoder_hidden)
            else:
                hidden_state = self.bidirTrans(encoder_hidden[0])
                cell_state = self.bidirTrans(encoder_hidden[1], True)
                encoder_hidden = (hidden_state, cell_state)
        logp, symbols = None, None
        if beam_search:
            logp, symbols, attns = self.decoder.summarize_bs(encoder_hidden,
                    self.max_title_len, encoder_output, input_lens)
        else:
            logp, symbols, attns = self.decoder.summarize(encoder_hidden,
                    self.max_title_len, encoder_output, input_lens)
        return logp, symbols, attns
