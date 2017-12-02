import heapq
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Attn import *
from .PointerNet import *

class DecoderRNN(nn.Module):
    def __init__(self, args, emb):
        super(DecoderRNN, self).__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.teach_ratio = args.teach_ratio
        self.attn_model = args.attn_model
        self.rnn_model = args.rnn_model
        self.use_copy = args.use_copy
        self.fix_pgen = args.fix_pgen
        self.use_separate_training = args.use_separate_training
        self.use_renorm = args.use_renorm
        self.renorm_method = args.renorm_method 

        self.sp_token_idx = {'EOS':1, 'UNK':2}   # global index for special tokens
        self.emb = emb
        emb_size = self.emb.weight.size(1)
        self.out_fc = nn.Linear(self.hidden_size, self.vocab_size)

        rnn_class = nn.GRU if self.rnn_model == 'gru' else nn.LSTM
        # extra hidden size due to feed input
        rnn_input_size = emb_size  + (0 if self.attn_model == 'none' else
                self.hidden_size)
        self.rnn = rnn_class(input_size = rnn_input_size,
                hidden_size = self.hidden_size, num_layers = args.nlayers,
                batch_first = True)

        if self.attn_model != 'none':
            concat_size = self.hidden_size * (3 if args.use_bidir else 2)
            self.concat = nn.Linear(concat_size, self.hidden_size)
            self.attn = Attn(args)
            if self.use_copy and self.fix_pgen < 0:
                self.ptr = PointerNet(args)

    def getAttnOutput(self, batch_input, last_output, h, encoder_output, inputs,
            input_lens, oov_size):
        input_emb = self.emb(batch_input)
        # feed input
        concat_input = torch.cat([input_emb, last_output], 1).unsqueeze(1)

        rnn_output, h = self.rnn(concat_input, h)
        attn_weights = self.attn(rnn_output, encoder_output, input_lens) # b x 1 x s
        context = attn_weights.bmm(encoder_output)

        # Final output layer (next word prediction) using the RNN hidden state
        # and context vector
        concat_input = torch.cat((rnn_output, context), 2).squeeze(1)
        concat_output = F.tanh(self.concat(concat_input))
        p_gen = None

        if self.use_copy:
            p_vocab = F.softmax(self.out_fc(concat_output))
            p_logp, p_gen = self.getPointerOutput(p_vocab, context,
                    attn_weights, input_emb, rnn_output, inputs, input_lens, oov_size)
        else:
            p_logp = F.log_softmax(self.out_fc(concat_output))
        return p_logp, h, concat_output, attn_weights, p_gen

    def getPointerOutput(self, p_vocab, context, attn_weights, input_emb,
            rnn_output, inputs_raw, input_lens, oov_size):
        """
        p_vocab: B x V
        inputs_raw: indexed inputs without replacing oov to UNK
        attn_weights: B x 1 x S
        """
        use_cuda = context.is_cuda
        batch_size = attn_weights.size(0)
        extVocab_size = self.vocab_size + oov_size

        if self.fix_pgen < 0:
            p_gen = self.ptr(context, rnn_output, input_emb)  # B x 1
        else:
            if self.training and self.use_separate_training:
                ratio = 0.9999 # apply this ratio to avoid inf loss
                use_gen_probabilty = random.random() < self.fix_pgen
                ones = Variable(torch.ones(batch_size, 1))
                p_gen = ones * (ratio if use_gen_probabilty else 1 - ratio)
            else:
                p_gen = Variable(torch.ones(batch_size, 1) * self.fix_pgen)
            p_gen = p_gen.cuda() if use_cuda else p_gen

        # attention re-normalization if instance contains oov
        attn_weights = attn_weights.squeeze(1) # b x seq_len
        if self.use_renorm:
            attn_weights, has_no_oov = self.renorm(attn_weights, inputs_raw, input_lens)
            # p_gen = torch.max(p_gen, has_no_oov)
            # print("p_gen after renorm: ", p_gen.cpu().data.numpy())

        # compute probability to generate from fix-sized vocabulary: p(gen) * P(w)
        p_gen_vocab = p_gen * p_vocab
        if oov_size > 0:
            p_gen_oov = Variable(torch.zeros(batch_size, oov_size))
            p_gen_oov = p_gen_oov.cuda() if use_cuda else p_gen_oov
        p_extVocab = torch.cat([p_gen_vocab, p_gen_oov], 1) if oov_size else p_gen_vocab   # B x ExtV

        # compute probability to copy from source: (1 - p(gen)) * P(w)
        p_copy_src = (1 - p_gen) * attn_weights
        p_extVocab.scatter_add_(1, Variable(inputs_raw), p_copy_src)
        return p_extVocab, p_gen

    def renorm(self, attn_weights, inputs_raw, input_lens):
        mask = Variable(inputs_raw >= self.vocab_size).float()
        masked_attn = (attn_weights * mask).add_(1e-10) # add small delta to avoid nan

        if self.renorm_method == 'div':
            # use division for re-normaliztion and get new attention weights
            attn_renorm = masked_attn / masked_attn.sum(dim=1, keepdim=True)
        elif self.renorm_method == 'softmax':
            # use softmax for re-normaliztion and get new attention weights
            batch_size, max_seq_len = inputs_raw.size()
            for b in range(batch_size):
                if input_lens[b] < max_seq_len:
                    masked_attn[b, input_lens[b]:] = float('-inf')
 
            attn_renorm = F.softmax(masked_attn)
            # print('attn_ori:', type(attn_weights), attn_weights.cpu().data.numpy())
            # print('attn_renorm: ', type(attn_renorm), attn_renorm.cpu().data.numpy())

        has_oov, _ = torch.max(mask, 1, keepdim=True)
        return attn_renorm, 1 - has_oov

    def getRNNOutput(self, batch_input, h):
        input_emb = self.emb(batch_input).unsqueeze(1) # b x 1 x hdim
        rnn_output, h = self.rnn(input_emb, h)
        xout = self.out_fc(rnn_output).squeeze(1)
        logp = F.log_softmax(xout)
        return logp, h

    def initLastOutput(self, batch_size):
        last_output = Variable(torch.zeros(batch_size, self.hidden_size))
        if next(self.parameters()).data.is_cuda:
            last_output = last_output.cuda()
        return last_output

    def forward(self, targets, h, encoder_output, inputs, input_lens, oov_size,
            force_scheduled_sampling):
        """
        targets: LongTensor, b x s
        """
        batch_size, max_seq_len = targets.size()
        batch_input = Variable(targets[:, 0]) #SOS, b
        batch_output, batch_p_gens = [], []
        use_teacher_forcing = random.random() < self.teach_ratio
        if force_scheduled_sampling:
            use_teacher_forcing = False
        if self.attn_model != 'none':
            last_output = self.initLastOutput(batch_size)

        for t in range(1, max_seq_len):
            if self.attn_model == 'none':
                p_logp, h = self.getRNNOutput(batch_input, h)
            else:
                p_logp, h, last_output, _, p_gen = self.getAttnOutput(
                        batch_input, last_output, h, encoder_output, inputs,
                        input_lens, oov_size)
                batch_p_gens.append(p_gen)

            batch_output.append(p_logp)
            if use_teacher_forcing:
                batch_input = Variable(targets[: ,t])
            else:
                _, batch_input = torch.max(p_logp, 1, keepdim=False)
            if self.use_copy:
                batch_input[batch_input >= self.vocab_size] = self.sp_token_idx['UNK']
        return batch_output, batch_p_gens

    def summarize(self, h, max_seq_len, encoder_output, inputs, input_lens,
            oov_size):
        """
        encoder_ouput: b x s x h (2h)
        """
        batch_size = encoder_output.size(0)
        last_output = self.initLastOutput(batch_size)
        # here it's assuming SOS has index 0
        batch_input = Variable(torch.zeros(batch_size).long())
        if next(self.parameters()).data.is_cuda:
            batch_input = batch_input.cuda()
        batch_attn, batch_p_gen, batch_symbol = [], [], [batch_input]

        for t in range(1, max_seq_len):
            if self.attn_model == 'none':
                p_logp, h = self.getRNNOutput(batch_input, h)
            else:
                p_logp, h, last_output, attn_weights, p_gen = self.getAttnOutput(
                        batch_input, last_output, h, encoder_output, inputs,
                        input_lens, oov_size)
                batch_attn.append(attn_weights.squeeze(1))
                batch_p_gen.append(p_gen)

            _, batch_input = torch.max(p_logp, 1, keepdim=False)
            batch_symbol.append(batch_input.clone())
            if self.use_copy:
                batch_input[batch_input >= self.vocab_size] = self.sp_token_idx['UNK']

        return batch_symbol, batch_attn, batch_p_gen

    def summarize_bs(self, h, max_seq_len, encoder_output, inputs, input_lens, 
            oov_size, beam_size=4):
        last_output = self.initLastOutput(1) # batch size is always 1
        # candidates heap: key: last_logp 
        #   value: (last_word, prev_words, outputs, h, last_output, prev_attns, prev_p_gens)
        last_candidates = [(0.0 ,
            (np.int64(0), [np.int64(0)], [0.0], h, last_output, [], []))]
        final_candidates = []

        current_depth = 0
        while last_candidates and current_depth < max_seq_len:
            current_depth += 1
            partial_candidates = []
            for last_logp, (last_word, prev_words, outputs, h, last_output, prev_attns, prev_p_gens) in last_candidates:
                # print(last_logp, last_word, prev_words)
                if last_word.item() == self.sp_token_idx['EOS']: 
                    while len(final_candidates) >= beam_size and last_logp > final_candidates[0][0]:
                        heapq.heappop(final_candidates)
                    if len(final_candidates) < beam_size:
                        heapq.heappush(final_candidates, (last_logp, prev_words, prev_attns, prev_p_gens))
                    continue
                if final_candidates and last_logp < final_candidates[0][0]:
                    continue

                inp = Variable(torch.Tensor.long(torch.zeros(1)).fill_(last_word.item()))
                if next(self.parameters()).data.is_cuda:
                    inp = inp.cuda()
                if self.attn_model == 'none':
                    logp, h = self.getRNNOutput(inp, h)
                    attn_weights, p_gen = None, None
                else:
                    logp, h, last_output, attn_weights, p_gen = self.getAttnOutput(
                            inp, last_output, h, encoder_output, inputs,
                            input_lens, oov_size)
                res, ind = logp.topk(beam_size)
                for i in range(ind.size(1)):
                    word = ind[0][i]
                    current_logp = last_logp + logp.data.numpy()[0][word.data.numpy()[0]]
                    while len(partial_candidates) + 1 > beam_size and current_logp > partial_candidates[0][0]:
                        heapq.heappop(partial_candidates)

                    if len(partial_candidates) + 1 <= beam_size:
                        heapq.heappush(partial_candidates, (current_logp,
                            (word.data.numpy()[0], prev_words+[word.data.numpy()[0]], outputs+[current_logp], 
                                h, last_output, prev_attns + [attn_weights], prev_p_gens + [p_gen])))
                    
            last_candidates = partial_candidates

        if final_candidates:
            last_logp, result_sent, prev_attns, prev_p_gens = max(final_candidates)
        else:
            last_logp, (_, result_sent, outputs, _, _, prev_attns, prev_p_gens) = max(last_candidates)
        symbol = []
        for result in result_sent:
            symbol.append(Variable(torch.Tensor.long(torch.zeros(1)).fill_(result.item())))
        return symbol, prev_attns, prev_p_gens
