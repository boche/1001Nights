import torch
import torch.nn as nn
import torch.nn.functional as F
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from util import *

class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.use_bidir = args.use_bidir
        self.rnn_model = args.rnn_model
        self.use_copy = args.use_copy
        self.max_title_len = args.max_title_len

        if self.use_bidir:
            self.linear_hidden = nn.Linear(args.hidden_size * 2, args.hidden_size)
            if self.rnn_model == 'lstm':
                self.linear_cell = nn.Linear(args.hidden_size * 2, args.hidden_size)

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(args, self.emb)
        self.decoder = DecoderRNN(args, self.emb)

    def bidirTrans(self, encoder_state, isCell=False):
        encoder_state = torch.cat([encoder_state[0::2, :, :],
            encoder_state[1::2, :, :]], 2)
        linear_func = self.linear_cell if isCell else self.linear_hidden
        return linear_func(encoder_state)

    def forward(self, inputs, input_lens, targets, target_lens, oov_size,
            is_volatile):
        encoder_output, encoder_hidden = self.encoder(inputs.clone(),
                input_lens, is_volatile)

        if self.use_bidir:
            if self.rnn_model == 'gru':
                encoder_hidden = self.bidirTrans(encoder_hidden)
            else:
                hidden_state = self.bidirTrans(encoder_hidden[0])
                cell_state = self.bidirTrans(encoder_hidden[1], True)
                encoder_hidden = (hidden_state, cell_state)

        p_logp, p_gen = self.decoder(targets.clone(), encoder_hidden,
                encoder_output, inputs.clone(), input_lens, oov_size)
        # decoder returns p if use_copy, otherwise logp
        mask_loss_func = p_mask_loss if self.use_copy else logp_mask_loss
        loss = mask_loss_func(p_logp, target_lens, targets)
        return loss, p_gen

    def summarize(self, inputs, input_lens, oov_size, beam_search, is_volatile):
        encoder_output, encoder_hidden = self.encoder(inputs.clone(),
                input_lens, is_volatile)
        if self.use_bidir:
            if self.rnn_model == 'gru':
                encoder_hidden = self.bidirTrans(encoder_hidden)
            else:
                hidden_state = self.bidirTrans(encoder_hidden[0])
                cell_state = self.bidirTrans(encoder_hidden[1], True)
                encoder_hidden = (hidden_state, cell_state)
        summarize_func = self.decoder.summarize_bs if beam_search else (
                self.decoder.summarize)
        return summarize_func(encoder_hidden, self.max_title_len,
                encoder_output, inputs.clone(), input_lens, oov_size)
