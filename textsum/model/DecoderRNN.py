import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import heapq

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
        batch_size, max_seq_len = target.size()
        use_teacher_forcing = random.random() < self.teach_ratio

        h = encoder_hidden
        batch_input = Variable(target[:, 0]) #SOS, b
        batch_output = []

        for t in range(1, max_seq_len):
            input_emb = self.emb(batch_input).unsqueeze(1) # b x 1 x hdim
            rnn_output, h = self.rnn(input_emb, h)
            xout = self.out(rnn_output).squeeze(1)
            logp = F.log_softmax(xout)
            batch_output.append(logp)

            if use_teacher_forcing:
                batch_input = Variable(target[: ,t])
            else:
                _, batch_input = torch.max(logp, 1, keepdim=False)
        return batch_output

    def summarize(self, encoder_hidden, max_seq_len):
        batch_size = encoder_hidden.size(1)
        h = encoder_hidden
        # here it's assuming SOS has index 0
        batch_input = Variable(torch.Tensor.long(torch.zeros(batch_size)))
        use_cuda = next(self.parameters()).data.is_cuda
        if use_cuda:
            batch_input = batch_input.cuda()
        batch_output = []
        batch_symbol = [batch_input]

        for t in range(1, max_seq_len):
            input_emb = self.emb(batch_input).unsqueeze(1)
            rnn_output, h = self.rnn(input_emb, h)
            xout = self.out(rnn_output).squeeze(1)
            logp = F.log_softmax(xout)
            batch_output.append(logp)

            _, batch_input = torch.max(logp, 1, keepdim=False)
            batch_symbol.append(batch_input)
        return batch_output, batch_symbol
    
    def summarize_bs(self, encoder_hidden, max_seq_len, beam_size=4):
        h = encoder_hidden
        
        def find_candidates(last_logp, last_word, prev_words, outputs, h):
            if last_word.item() == 1: #EOS
                while len(final_candidates) >= beam_size and last_logp > final_candidates[0][0]:
                    heapq.heappop(final_candidates)
                if len(final_candidates) < beam_size:
                    heapq.heappush(final_candidates, (last_logp, prev_words))
                return
            if final_candidates and last_logp < final_candidates[0][0]:
                return

            inp = Variable(torch.Tensor.long(torch.zeros(1)).fill_(last_word.item()))
            input_emb = self.emb(inp).unsqueeze(1)
            rnn_output, h = self.rnn(input_emb, h)
            xout = self.out(rnn_output).squeeze(1)
            logp = F.log_softmax(xout)
            res, ind = logp.topk(beam_size)
            for i in range(ind.size(1)):
                word = ind[0][i]
                # if word.data.numpy()[0] == 2:
                #    continue
                current_logp = last_logp + logp.data.numpy()[0][word.data.numpy()[0]]
                while len(partial_candidates) + 1 > beam_size and current_logp > partial_candidates[0][0]:
                    heapq.heappop(partial_candidates)

                if len(partial_candidates) + 1 <= beam_size:
                    heapq.heappush(partial_candidates, (current_logp, (word.data.numpy()[0], prev_words+[word.data.numpy()[0]], outputs+[current_logp], h)))

        last_candidates = [(0.0 ,(np.int64(0), [np.int64(0)], [0.0], h))]
        final_candidates = []

        current_depth = 0
        while last_candidates and current_depth < max_seq_len:
            current_depth += 1
            partial_candidates = []
            # print('***********')
            for last_logp, (last_word, prev_words, outputs, h) in last_candidates:
                # print(last_logp, last_word, prev_words)
                find_candidates(last_logp, last_word, prev_words, outputs, h)
            last_candidates = partial_candidates

        if final_candidates:
            last_logp, result_sent = max(final_candidates)
        else:                     
            last_logp, (_, result_sent, outputs, _) = max(last_candidates)
        symbol = []
        for result in result_sent:
            symbol.append(Variable(torch.Tensor.long(torch.zeros(1)).fill_(result.item())))
        return outputs, symbol
