import torch
import torch.nn as nn
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from util import *

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
        self.use_pointer_net = args.use_pointer_net

        # encoder and decoder share a common embedding layer
        self.emb = nn.Embedding(args.vocab_size, args.emb_size)
        self.encoder = EncoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.dropout, self.rnn_model, self.use_pointer_net)
        self.decoder = DecoderRNN(self.vocab_size, self.emb, self.hidden_size,
                self.nlayers, self.teach_ratio, self.dropout, self.rnn_model,
                self.use_pointer_net, self.attn_model)

    def forward(self, inputs_ori, input_lens, targets_ori, target_lens, oov_size):
        inputs, targets = inputs_ori.clone(), targets_ori.clone()
        inputs_raw = inputs_ori.clone()
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        p_logp, p_gen = self.decoder(targets, encoder_hidden, encoder_output, inputs_raw, input_lens, oov_size)

        mask_loss_func = p_mask_loss if self.use_pointer_net else logp_mask_loss
        loss = mask_loss_func(p_logp, target_lens, targets_ori)
        return loss, p_gen

    def summarize(self, inputs_ori, input_lens, oov_size, beam_search=True):
        inputs, inputs_raw = inputs_ori.clone(), inputs_ori.clone()
        encoder_output, encoder_hidden = self.encoder(inputs, input_lens)
        logp, symbols = None, None
        if beam_search:
            logp, symbols, attns = self.decoder.summarize_bs(encoder_hidden,
                    self.max_title_len, encoder_output, input_lens)
        else:
            logp, symbols, attns, p_gens = self.decoder.summarize(encoder_hidden,
                    self.max_title_len, encoder_output, inputs_raw, input_lens, oov_size)
        return logp, symbols, attns, p_gens
