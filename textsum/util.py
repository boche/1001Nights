import numpy as np
import torch
from torch.autograd import Variable
from tokens import *
import matplotlib as mpl
mpl.use('Agg') #adding this because otherwise plt will fail because of no display
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def mask_loss(p_logp_list, target_lens, targets, is_logp):
    """
    p_logp_list: list of torch tensors, (seq_len - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq_len = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    target_lens = target_lens.cuda() if p_logp_list[0].is_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq_len - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        p_logp = torch.gather(p_logp_list[i], 1, idx).view(-1)
        logp = p_logp if is_logp else torch.log(p_logp)
        loss += logp[target_lens > i + 1].sum()
    return -loss

def mask_generation_prob(prob_list, target_lens):
    """
    prob_list: list of p_gens (not include prob for <SOS>); batch_size x 1
    target_lens: list; note that target_len includes <SOS> and <EOS> for each sentence
    """
    prob_sum = 0
    probs = torch.stack(prob_list).transpose(0, 1).data.cpu().numpy() # b x s
    for i, prob in enumerate(probs):
        prob_sum += prob[:target_lens[i] - 1].sum()
    return prob_sum

def visualize(input_text, output_text, gold_text, attn, p_gen, args):
    """
    attn: output_s x input_s
    """
    input_words = [''] + input_text.split(' ')
    output_words = [''] + output_text.split(' ') + [EOS]
    attn = attn.data.cpu().numpy()[:len(output_words) - 1, :]
    fig = plt.figure(figsize=(12, 8))

    if args.use_copy:
        gs = GridSpec(5, 4)

        ax_attn = plt.subplot(gs[:, :-1])
        cax_attn = ax_attn.matshow(attn, cmap='bone')
        fig.colorbar(cax_attn, ax=ax_attn, orientation='horizontal')
        ax_attn.set_title('attention scores')
        ax_attn.set_xticklabels(input_words, rotation=90)
        ax_attn.set_yticklabels(output_words)
        ax_attn.xaxis.set_ticks_position('bottom')
        ax_attn.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_attn.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax_prob = plt.subplot(gs[:, -1:])
        p_gen = p_gen.data.cpu().numpy()[:len(output_words) - 1, :]
        cax_prob = ax_prob.matshow(p_gen, cmap='bone')
        fig.colorbar(cax_prob, ax=ax_prob, orientation='vertical')
        ax_prob.set_title('p_gen')
        ax_prob.set_xticks([])
        ax_prob.set_xticklabels([])
        ax_prob.set_yticklabels(output_words)
        ax_prob.yaxis.set_major_locator(ticker.MultipleLocator(1))
    else:
        ax = fig.add_subplot(111)
        cax = ax.matshow(attn, cmap='bone')
        fig.colorbar(cax, orientation='horizontal')

        # Set up axes
        ax.set_xticklabels(input_words, rotation=90)
        ax.set_yticklabels(output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig("%s/figure/attn/%s.png" % (args.user_dir,
        gold_text.replace(" ", "_").replace('/', '_')), dpi = 200)
    plt.close()

def pad_seq(seq, max_length):
    seq += [0] * (max_length - len(seq))
    return seq

def group_data(data, word2idx, args):
    """
    data: list of (docid, head, body)
    """
    # sort by input length, group inputs with close length in a batch
    sorted_data = sorted(data, key = lambda x: len(x[1]), reverse = True)
    nbatches = (len(data) + args.batch_size - 1) // args.batch_size
    return [next_batch(i, sorted_data, word2idx, args) for i in range(nbatches)]

def next_batch(batch_idx, data, word2idx, args):
    targets, inputs = [], []
    start = batch_idx * args.batch_size
    end = min(len(data), start + args.batch_size)

    batch_data = sorted(data[start:end], key = lambda x: len(x[2]), reverse = True)
    # preprocessing should already discard empty documents
    for docid, head, body in batch_data:
        inputs.append(body[:args.max_text_len])
        targets.append([word2idx[SOS]] + head[:args.max_title_len] + [word2idx[EOS]])

    # create a padded sequence
    input_lens = [len(x) for x in inputs]
    inputs = [pad_seq(x, max(input_lens)) for x in inputs]
    target_lens = [len(y) for y in targets]
    targets = [pad_seq(y, max(target_lens)) for y in targets]
    return inputs, targets, input_lens, target_lens

def build_local_dict(inputs, targets, word2idx, args):
    """
    inputs: list of index-text hybrid sequence for body (Eg: [92, EMP, 2, 78])
    targets: same as above, but for headline.
    """
    inps, tgts = [], []
    loc_word2idx, loc_idx2word = {}, {}
    loc_idx = args.vocab_size # size of global index

    for inp, tgt in zip(inputs, targets):
        tempInp, tempTgt = [], []
        instance_oov = set() # hash set for oovs in a single datapoint instance
        for word in inp:
            if isinstance(word, str): # an out-of-vocabulary word
                if args.use_copy:
                    instance_oov.add(word)
                    if word not in loc_word2idx:
                        loc_word2idx[word] = loc_idx
                        loc_idx2word[loc_idx] = word
                        loc_idx += 1
                    tempInp.append(loc_word2idx[word])
                else:
                    # if not using copy, replace oov with UNK
                    tempInp.append(word2idx[UNK])
            else:
                tempInp.append(word)

        for word in tgt:
            """
            In copy mode, an oov word that only exists in target will transform
            to UNK. Otherwise, instance_oov should be empty, so all oov will be
            replaced with UNK.
            """
            if isinstance(word, str):
                tempTgt.append(loc_word2idx[word] if word in instance_oov else word2idx[UNK])
            else:
                tempTgt.append(word)

        inps.append(tempInp)
        tgts.append(tempTgt)
    return inps, tgts, loc_word2idx, loc_idx2word

def index_oov(inputs, targets, word2idx, args):
    loc_word2idx, loc_idx2word = {}, {}
    inputs, targets, loc_word2idx, loc_idx2word = build_local_dict(inputs,
            targets, word2idx, args)
    inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
    if args.use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    return inputs, targets, loc_word2idx, loc_idx2word

def idxes2sent(idxes, idx2word, loc_idx2word, keepSrc):
    seq = []
    for idx in idxes.numpy():
        if idx < len(idx2word):
            if idx2word[idx] == SOS: continue
            if idx2word[idx] == EOS: break
            seq.append(idx2word[idx])
        else:
            seq.append(("" if keepSrc else "[COPY]_") + loc_idx2word[idx])

    # some characters may not be printable if not encode by utf-8
    return " ".join(seq).encode('utf-8').decode("utf-8")

def check_args(args):
    if args.use_copy and args.attn_model == 'none':
        raise Exception('Attention model should not be none when using copy!')
    if args.fix_pgen >= 0 and not args.use_copy:
        raise Exception('Fix pgen can only be enabled when using copy!')
    if args.use_separate_training and not args.use_copy:
        raise Exception('Separate training can only be enabled when using copy!')
    if args.use_bidir and args.attn_model == 'dot':
        raise Exception("Bidirectional encoder doesn't work with dot attention!")

def vectorize(raw_data, word2idx, args):
    data_vec = []
    keepOOV = args.use_copy
    for data in raw_data:
        docid, headline, body = data
        headline = [word2idx.get(w, w if keepOOV else word2idx[UNK]) for w in headline]
        headline = [word2idx[SOS]] + headline + [word2idx[EOS]]
        body = [word2idx.get(w, w if keepOOV else word2idx[UNK]) for w in body[:args.max_text_len]]
        data_vec.append((docid, headline, body))
    return data_vec
