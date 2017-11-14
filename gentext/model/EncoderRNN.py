import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, emb, hidden_size, word_layers, sent_layers, dropout,
            rnn_model):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.emb = emb
        self.dropout = nn.Dropout(dropout)
        emb_size = self.emb.weight.size(1)

        rnn_class = nn.GRU if rnn_model == 'gru' else nn.LSTM
        self.word_rnn = rnn_class(input_size=emb_size, hidden_size = hidden_size,
                dropout = dropout, num_layers = word_layers, batch_first = True)
        self.sent_rnn = rnn_class(input_size=hidden_size, hidden_size=hidden_size,
                dropout = dropout, num_layers = sent_layers, batch_first = True)

    def forward(self, inputs, inputs_len, inputs_eos_indices, is_volatile):
        """
        inputs: batch x seq_len, word index, LongTensor
        inputs_len: batch, list
        inputs_eos_indices: batch x nsent, LongTensor
        """
        word_emb = self.dropout(self.emb(
            Variable(inputs, volatile = is_volatile))) # -> b x s x e
        word_pack = nn.utils.rnn.pack_padded_sequence(word_emb, inputs_len,
                batch_first=True)
        word_res_pack, word_state = self.word_rnn(word_pack)
        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_res_pack,
                batch_first=True)
        # word_output: b x s x h
        # word_state: nwl x b x h, tuple if lstm

        # gather the end output for each sentence
        gather_indices = Variable(inputs_eos_indices.unsqueeze(2).repeat(
                1, 1, self.hidden_size)) # b x ns x h
        sent_emb = word_output.gather(1, gather_indices) # -> b x ns x h
        sent_output, sent_state = self.sent_rnn(sent_emb)
        # sent_output: b x ns x h
        # sent_state: nsl x b x h, tuple if lstm
        return word_output, word_state, sent_output, sent_state
