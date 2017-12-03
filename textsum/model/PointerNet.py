import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, args):
        super(PointerNet, self).__init__()
        generator_input_size = args.emb_size + args.hidden_size * (
                3 if args.use_bidir else 2)
        generator_input_size += args.hidden_size if args.use_decoder_attn else 0
        self.generator = nn.Linear(generator_input_size, 1, bias=True)

    def forward(self, context, rnn_output, input_emb):
        context = context.squeeze(1)         # B x H, 2H if use_bidir
        rnn_output = rnn_output.squeeze(1)   # B x H
        gen_input = torch.cat([input_emb, context, rnn_output], 1)
        prob_gen = F.sigmoid(self.generator(gen_input))
        return prob_gen
