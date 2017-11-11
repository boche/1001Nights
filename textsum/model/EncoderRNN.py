import torch
import torch.nn as nn
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb, hidden_size, nlayers, dropout, rnn_model, use_pointer_net):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        # self.dropout = nn.Dropout(dropout)
        self.emb = emb
        self.use_pointer_net = use_pointer_net
        
        emb_size = self.emb.weight.size(1)
        rnn_class = nn.GRU if rnn_model == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_size=emb_size, hidden_size = hidden_size, 
                dropout = dropout, num_layers = nlayers, batch_first = True)

    def forward(self, inputs, input_lens):
        """
        inputs: batch x seq_len
        """
        # inputs_emb = self.dropout(self.emb(Variable(inputs)))
        # set oov words to <UNK> (index = 2) for encoder
        if self.use_pointer_net:
            inputs[inputs >= self.vocab_size] = 2
        
        inputs_emb = self.emb(Variable(inputs))
        inputs_pack = nn.utils.rnn.pack_padded_sequence(inputs_emb, input_lens, batch_first=True)
        output_pack, hidden = self.rnn(inputs_pack)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_pack, batch_first=True)
        return output, hidden
