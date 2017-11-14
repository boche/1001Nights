import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from util import mask_loss

class WordDecoderRNN(nn.Module):
    def __init__(self, emb, hidden_size, nlayers, dropout, rnn_model):
        super(WordDecoderRNN, self).__init__()
        vocab_size, emb_size = emb.weight.size()

        self.emb = emb
        self.dropout = nn.Dropout(dropout)
        rnn_class = nn.GRU if rnn_model == 'gru' else nn.LSTM
        self.word_rnn = rnn_class(input_size = 2 * emb_size, hidden_size = hidden_size,
                dropout = dropout, num_layers = nlayers, batch_first = True)
        self.out_fc = nn.Linear(hidden_size, vocab_size)
        self.SOS_IDX = 0 # assume sos idx is 0
        self.EOS_IDX = 1 # assume eos idx is 1

    def forward(self, targets, targets_kws, word_state, targets_len):
        """
        targets: batch x seq_len, word index, LongTensor
        targets_kws: batch x seq_len, keyword index, LongTensor
        word_state: nl x b x h, tuple if lstm
        """
        seq_len = targets.size(1)
        logp_list = []

        for i in range(seq_len - 1):
            last_word = Variable(targets[:, i])
            keyword = Variable(targets_kws[:, i+1])
            input_emb = self.dropout(torch.cat((self.emb(keyword), self.emb(
                last_word)), 1)).unsqueeze(1) # -> b x 1 x 2e
            word_output, word_state = self.word_rnn(input_emb, word_state)
            # word_output: b x 1 x h
            logp =  F.log_softmax(self.out_fc(word_output.squeeze(1))) # b x v
            logp_list.append(logp)
        return mask_loss(logp_list, targets_len, targets)

    def sample(self, word_state, max_sent_len, target_kws):
        """
        sample per instance
        word_state: nl x 1 x h
        target_kws: 1 x nsent
        """
        res = []
        nsent = target_kws.size(1)
        last_word = Variable(torch.LongTensor([self.SOS_IDX]))
        last_word = last_word.cuda() if word_state.is_cuda else last_word

        for s in range(nsent):
            keyword = Variable(target_kws[:, s])
            last_word_idx = 0
            for _ in range(max_sent_len):
                input_emb = self.dropout(torch.cat((self.emb(keyword), self.emb(
                    last_word)), 1)).unsqueeze(1) # -> 1 x 1 x 2e
                word_output, word_state = self.word_rnn(input_emb, word_state)
                logp =  F.log_softmax(self.out_fc(word_output.squeeze(1))) # 1xv
                _, last_word = logp.max(1)

                last_word_idx = int(last_word[0].data.cpu().numpy())
                res.append(last_word_idx)
                if last_word_idx == self.EOS_IDX:
                    break
            if last_word_idx != self.EOS_IDX:
                res.append(self.EOS_IDX)
        return res
