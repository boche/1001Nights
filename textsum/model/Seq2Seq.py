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
            self.linear_hidden = nn.Linear(self.hidden_size * 2, self.hidden_size)
            if self.rnn_model == 'lstm':
                self.linear_cell = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.use_pointer_net = args.use_pointer_net

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.dropout, self.rnn_model, self.use_pointer_net, self.bidir)
        self.decoder = DecoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.teach_ratio, self.dropout, self.rnn_model,
                self.use_pointer_net, self.bidir, self.attn_model)
    
    def bidirTrans(self, encoder_state, isCell=False):
        encoder_state = torch.cat([encoder_state[0::2, :, :], encoder_state[1::2, :, :]], 2)
        linear_func = self.linear_cell if isCell else self.linear_hidden
        return linear_func(encoder_state)

    def forward(self, inputs, input_lens, targets, oov_size):
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        if self.bidir:
            if self.rnn_model == 'gru':
                encoder_hidden = self.bidirTrans(encoder_hidden)
            else:
                hidden_state = self.bidirTrans(encoder_hidden[0])
                cell_state = self.bidirTrans(encoder_hidden[1], True)
                encoder_hidden = (hidden_state, cell_state)
        logp, p_gen = self.decoder(targets, encoder_hidden, encoder_output, inputs, input_lens, oov_size)
        return logp, p_gen

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
            logp, symbols, attns, p_gens = self.decoder.summarize(encoder_hidden,
                    self.max_title_len, encoder_output, inputs, input_lens)
        return logp, symbols, attns, p_gens
