import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, args, emb):
        super(EncoderRNN, self).__init__()
        self.vocab_size = args.vocab_size
        self.use_copy = args.use_copy
        self.IDX_UNK = 2  # global index for UNK
        self.emb = emb

        emb_size = self.emb.weight.size(1)
        rnn_class = nn.GRU if args.rnn_model == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=emb_size, hidden_size = args.hidden_size,
                num_layers = args.nlayers, bidirectional = args.use_bidir,
                batch_first = True)

    def forward(self, inputs, input_lens, is_volatile):
        """
        inputs: batch x seq_len
        """
        if self.use_copy:
            # set oov words to <UNK> for encoder
            inputs[inputs >= self.vocab_size] = self.IDX_UNK

        inputs_emb = self.emb(Variable(inputs, volatile = is_volatile))
        inputs_pack = nn.utils.rnn.pack_padded_sequence(inputs_emb, input_lens,
                batch_first=True)
        output_pack, hidden = self.rnn(inputs_pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_pack, batch_first=True)
        return output, hidden
