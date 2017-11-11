import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, emb, hidden_size, nlayers, dropout, rnn_model):
        super(DecoderRNN, self).__init__()
        vocab_size, emb_size = emb.weight.size()

        self.emb = emb
        self.dropout = nn.Dropout(dropout)
        rnn_class = nn.GRU if rnn_model == 'gru' else nn.LSTM
        self.word_rnn = rnn_class(input_size = 2 * emb_size, hidden_size = hidden_size,
                dropout = dropout, num_layers = nlayers, batch_first = True)
        self.sent_rnn = rnn_class(input_size = hidden_size, hidden_size=hidden_size,
                dropout = dropout, num_layers = nlayers, batch_first = True)
        self.out_fc = nn.Linear(hidden_size, vocab_size)
        self.SOS_IDX = 0 # assume sos idx is 0
        self.EOS_IDX = 1 #assuming eos idx is 1

    def forward(self, targets, targets_kws, sent_state, targets_len):
        """
        targets: batch x seq_len, word index, LongTensor
        targets_kws: batch x seq_len, keyword index, LongTensor
        sent_state: nl x b x h, tuple if lstm
        """
        batch_size, seq_len = targets.size()
        word_state = sent_state
        nl, _, hidden_size = sent_state.size()
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

            # only update sentence state and word state if current word is eos
            mask_eos = Variable((targets[:, i+1] == self.EOS_IDX).float().
                    unsqueeze(0).unsqueeze(2).repeat( nl, 1, hidden_size))
            # mask_eos: nl x b x h
            sent_output, tmp_sent_state = self.sent_rnn(word_output, sent_state)
            sent_state = sent_state * (1 - mask_eos) + tmp_sent_state * mask_eos
            word_state = word_state * (1 - mask_eos) + tmp_sent_state * mask_eos
        return mask_loss(logp_list, targets_len,targets)

    def sample(self, sent_state, max_sent_len, target_kws):
        """
        sample per instance
        sent_state: nl x 1 x h
        target_kws: 1 x nsent
        """
        res = []
        use_cuda = sent_state.is_cuda
        nsent = target_kws.size(1)
        last_word = Variable(torch.LongTensor([self.SOS_IDX])) 
        last_word = last_word.cuda() if use_cuda else last_word

        for s in range(nsent):
            word_state = sent_state
            keyword = Variable(target_kws[:, s])
            last_word_idx = 0
            for i in range(max_sent_len):
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
            sent_output, sent_state = self.sent_rnn(word_output, sent_state)
        return res

def mask_loss(logp_list, target_lens, targets):
    """
    logp_list: list of torch tensors, (seq_len - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq_len = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    use_cuda = logp_list[0].is_cuda
    target_lens = target_lens.cuda() if use_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq_len - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        logp = torch.gather(logp_list[i], 1, idx).view(-1)
        loss += logp[target_lens > i + 1].sum()
    return -loss
