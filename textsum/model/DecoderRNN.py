import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb, hidden_size, nlayers, teach_ratio):
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.teach_ratio = teach_ratio

        self.emb = emb
        emb_size = self.emb.weight.size(1)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size = hidden_size,
                num_layers = nlayers, batch_first = True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, target, encoder_hidden):
        """
        target: batch_size, max_seq_len
        """
        batch_size, max_seq_len = target.size()
        use_teacher_forcing = random.random() < self.teach_ratio

        h = encoder_hidden
        batch_input = target[:, 0] #SOS
        batch_output = []

        for t in range(1, max_seq_len):
            input_emb = self.emb(batch_input).unsqueeze(1)
            rnn_output, h = self.rnn(input_emb, h)
            xout = self.out(rnn_output)
            logp = F.log_softmax(xout).squeeze(1)
            batch_output.append(logp)

            if use_teacher_forcing:
                batch_input = target[: ,t]
            else:
                _, batch_input = torch.max(logp, 1, keepdim=False)
        return batch_output

    def summarize(self, encoder_hidden, max_seq_len):
        batch_size = encoder_hidden.size(1)

        h = encoder_hidden
        batch_input = Variable(torch.Tensor.long(torch.zeros(batch_size)).cuda())
        batch_output = []
        batch_symbol = [batch_input]

        for t in range(1, max_seq_len):
            input_emb = self.emb(batch_input).unsqueeze(1)
            rnn_output, h = self.rnn(input_emb, h)
            xout = self.out(rnn_output)
            logp = F.log_softmax(xout).squeeze(1)
            batch_output.append(logp)

            _, batch_input = torch.max(logp, 1, keepdim=False)
            batch_symbol.append(batch_input)
        return batch_output, batch_symbol
