import heapq
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .attn import *
from .pointerNet import *

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb, hidden_size, nlayers, teach_ratio,
            dropout, rnn_model, use_pointer_net, attn_model='general'):
        # attn_model supports: 'none', 'general', 'dot'
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.teach_ratio = teach_ratio
        self.attn_model = attn_model
        self.rnn_model = rnn_model
        self.use_pointer_net = use_pointer_net
        
        self.IDX_UNK = 2   # global index for <UNK>
        self.emb = emb
        # self.dropout = nn.Dropout(dropout)
        emb_size = self.emb.weight.size(1)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        rnn_class = nn.GRU if rnn_model == 'gru' else nn.LSTM
        rnn_input_size = emb_size if attn_model == 'none' else emb_size + hidden_size
        self.rnn = rnn_class(input_size = rnn_input_size, hidden_size = hidden_size,
                    dropout = dropout, num_layers = nlayers, batch_first = True)

        if self.attn_model != 'none':
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.attn_model = attn_model
            self.attn = Attn(attn_model, hidden_size)
            
            if self.use_pointer_net:  # activate copy mechanism from pointer net
                self.ptr = PointerNet(emb_size, hidden_size)

    def getAttnOutput(self, batch_input, last_output, h, encoder_output, inputs_raw, input_lens, oov_size):
        input_emb = self.emb(batch_input)
        concat_input = torch.cat([input_emb, last_output], 1).unsqueeze(1)

        rnn_output, h = self.rnn(concat_input, h)
        attn_weights = self.attn(rnn_output, encoder_output, input_lens) # b x 1 x s
        context = attn_weights.bmm(encoder_output)
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        concat_input = torch.cat((rnn_output, context), 2).squeeze(1)
        concat_output = F.tanh(self.concat(concat_input))
        p_gen = None
        
        if self.use_pointer_net:
            p_vocab = F.softmax(self.out(concat_output))
            logp, p_gen = self.getPointerOutput(p_vocab, context, attn_weights, input_emb, rnn_output, inputs_raw, oov_size)
        else:
            logp = F.log_softmax(self.out(concat_output))
        return logp, h, concat_output, attn_weights, p_gen
    

    def getPointerOutput(self, p_vocab, context, attn_weights, input_emb, rnn_output, inputs_raw, oov_size):
        """
        p_vocab: B x V
        inputs_raw: indexed inputs without replacing oov to UNK
        attn_weights: B x 1 x S
        """
        p_gen = self.ptr(context, rnn_output, input_emb)  # B x 1, broadcastable
        batch_size = p_gen.size(0)
        extVocab_size = self.output_size + self.oov_size
        self.use_cuda = p_gen.data.is_cuda
        
        # compute probability to generate from fix-sized vocabulary: p(gen) * P(w)
        p_gen_vocab = p_gen * p_vocab
        if self.oov_size != 0:
            p_gen_oov = Variable(torch.zeros(batch_size, self.oov_size))
            p_gen_oov = p_gen_oov.cuda() if self.use_cuda else p_gen_oov
        p_extVocab = torch.cat([p_gen_vocab, p_gen_oov], 1) if self.oov_size else p_gen_vocab   # B x ExtV
        
        # compute probability to copy from source: (1 - p(gen)) * P(w)
        p_copy_src = (1 - p_gen) * attn_weights.squeeze(1)
        p_extVocab.scatter_add_(1, Variable(inputs_raw), p_copy_src)
 
        if self.oov_size != oov_size:
            print("self.oov_size != oov_size !!!!!!!!!!!!!!1") 
        
        # # experiment for scatter_add()
        # x = Variable(torch.Tensor([[9,10,11,12], [2,3,0,0]]))
        # idx = Variable(torch.LongTensor([[4,3,4,7], [1,1,2,2]]))
        # out = Variable(torch.randn(2,8))
        # out.scatter_add_(1, idx, x)
        
        # # assert sum of probs of each instance is 1
        # p_sum = torch.sum(p_extVocab, 1)
        # print(p_sum.data.numpy())

        # print(attn_weights[13, :].squeeze(1).cpu().data.numpy())
        # print(inputs_raw[13, :].cpu().numpy())
        # print(p_extVocab.size())
        # print(p_extVocab[13, self.vocab_size:].cpu().data.numpy())
        # print(p_gen.cpu().data.numpy())
        return torch.log(p_extVocab), p_gen
    
    def getRNNOutput(self, batch_input, h):
        # input_emb = self.dropout(self.emb(batch_input).unsqueeze(1)) # b x 1 x hdim
        input_emb = self.emb(batch_input).unsqueeze(1) # b x 1 x hdim
        rnn_output, h = self.rnn(input_emb, h)
        xout = self.out(rnn_output).squeeze(1)
        logp = F.log_softmax(xout)
        return logp, h

    def initLastOutput(self, batch_size):
        last_output = Variable(torch.zeros(batch_size, self.hidden_size))
        if next(self.parameters()).data.is_cuda:
            last_output = last_output.cuda()
        return last_output
    
    def forward(self, target, encoder_hidden, encoder_output, inputs_raw, input_lens, oov_size):
        self.oov_size = oov_size
        batch_size, max_seq_len = target.size()
        h = encoder_hidden
        batch_input = Variable(target[:, 0]) #SOS, b
        batch_output, batch_p_gens = [], []
        
        if self.attn_model != 'none':
            last_output = self.initLastOutput(batch_size)
        use_teacher_forcing = random.random() < self.teach_ratio

        for t in range(1, max_seq_len):
            if self.attn_model == 'none':
                logp, h = self.getRNNOutput(batch_input, h)
            else:
                logp, h, last_output, _, p_gen = self.getAttnOutput(batch_input, last_output,
                                              h, encoder_output, inputs_raw, input_lens, oov_size)
            
            batch_output.append(logp)
            batch_p_gens.append(p_gen)
            if use_teacher_forcing:
                batch_input = Variable(target[: ,t])
                if self.use_pointer_net:
                    # set oov words to <UNK> for decoder input
                    batch_input[batch_input >= self.vocab_size] = self.IDX_UNK
            else:
                _, batch_input = torch.max(logp, 1, keepdim=False)
                # TO DO: sample for pointer net (if sampled word is oov)
                
        return batch_output, batch_p_gens

    def summarize(self, encoder_hidden, max_seq_len, encoder_output, inputs_raw, input_lens, oov_size):
        batch_size = encoder_hidden.size(1) if self.rnn_model == 'gru' else encoder_hidden[0].size(1)
        h = encoder_hidden
        # here it's assuming SOS has index 0
        batch_input = Variable(torch.Tensor.long(torch.zeros(batch_size)))
        use_cuda = next(self.parameters()).data.is_cuda
        if use_cuda:
            batch_input = batch_input.cuda()
        batch_output, batch_attn, batch_p_gen = [], [], []
        batch_symbol, p_gen = [batch_input], None
        last_output = self.initLastOutput(batch_size)

        for t in range(1, max_seq_len):
            if self.attn_model == 'none':
                logp, h = self.getRNNOutput(batch_input, h)
            else:
                logp, h, last_output, attn_weights, p_gen = self.getAttnOutput(
                        batch_input, last_output, h, encoder_output, inputs_raw, input_lens, oov_size)
                batch_attn.append(attn_weights.squeeze(1))
            batch_output.append(logp)
            batch_p_gen.append(p_gen)
            
            # print('logp: ', logp.size())
            batch_max_logp, batch_input = torch.max(logp, 1, keepdim=False)
            
            #if t == max_seq_len // 3:
                # print(type(batch_max_logp), batch_max_logp.size())
                # print(type(batch_input), batch_input.data.cpu().numpy())
                # print(torch.cat([batch_max_logp.unsqueeze(1), batch_input.float().unsqueeze(1)], 1).data.cpu().numpy())
                # print(logp[13, self.vocab_size:].data.cpu().numpy())           
            
            batch_symbol.append(batch_input.clone())
            if self.use_pointer_net:
                batch_input[batch_input >= self.vocab_size] = self.IDX_UNK

        return batch_output, batch_symbol, batch_attn, batch_p_gen

    def summarize_bs(self, encoder_hidden, max_seq_len, encoder_output, input_lens, beam_size=4):
        batch_size = encoder_hidden.size(1) if self.rnn_model == 'gru' else encoder_hidden[0].size(1)

        h = encoder_hidden
        last_output = self.initLastOutput(batch_size)
        last_candidates = [(0.0 ,(np.int64(0), [np.int64(0)], [0.0], h, last_output))]

        def find_candidates(last_logp, last_word, prev_words, outputs, h, last_output):
            if last_word.item() == 1: #EOS
                while len(final_candidates) >= beam_size and last_logp > final_candidates[0][0]:
                    heapq.heappop(final_candidates)
                if len(final_candidates) < beam_size:
                    heapq.heappush(final_candidates, (last_logp, prev_words))
                return
            if final_candidates and last_logp < final_candidates[0][0]:
                return

            inp = Variable(torch.Tensor.long(torch.zeros(1)).fill_(last_word.item()))
            if self.attn_model == 'none':
                logp, h = self.getRNNOutput(inp, h)
            else:
                logp, h, last_output, _ = self.getAttnOutput(inp, last_output, h, encoder_output, input_lens)
            res, ind = logp.topk(beam_size)
            for i in range(ind.size(1)):
                word = ind[0][i]
                # if word.data.numpy()[0] == self.IDX_UNK: # skip if it's UNK
                #    continue
                current_logp = last_logp + logp.data.numpy()[0][word.data.numpy()[0]]
                while len(partial_candidates) + 1 > beam_size and current_logp > partial_candidates[0][0]:
                    heapq.heappop(partial_candidates)

                if len(partial_candidates) + 1 <= beam_size:
                    heapq.heappush(partial_candidates, (current_logp,
                        (word.data.numpy()[0], prev_words+[word.data.numpy()[0]], outputs+[current_logp], h, last_output)))

        final_candidates = []

        current_depth = 0
        while last_candidates and current_depth < max_seq_len:
            current_depth += 1
            partial_candidates = []
            for last_logp, (last_word, prev_words, outputs, h, last_output) in last_candidates:
                # print(last_logp, last_word, prev_words)
                find_candidates(last_logp, last_word, prev_words, outputs, h, last_output)
            last_candidates = partial_candidates

        if final_candidates:
            last_logp, result_sent = max(final_candidates)
        else:
            last_logp, (_, result_sent, outputs, _, _) = max(last_candidates)
        symbol = []
        for result in result_sent:
            symbol.append(Variable(torch.Tensor.long(torch.zeros(1)).fill_(result.item())))
        return outputs, symbol, None # we don't plot attn for beam search
