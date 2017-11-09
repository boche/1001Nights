import sys
import argparse
import random
import string
import numpy as np
import time
import pickle
from tokens import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Seq2Seq
import matplotlib as mpl
mpl.use('Agg') #adding this because otherwise plt will fail because of no display
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from rouge import Rouge

def pad_seq(seq, max_length):
    seq += [0] * (max_length - len(seq))
    return seq

def group_data(data):
    """
    data: list of (docid, head, body)
    """
    group_data = []
    # sort by input length, group inputs with close length in a batch
    sorted_data = sorted(data, key = lambda x: len(x[2]), reverse = True)
    nbatches = (len(data) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(nbatches):
        group_data.append(next_batch(batch_idx, sorted_data))
    return group_data

def next_batch(batch_idx, data):
    targets, inputs = [], []
    start = batch_idx * args.batch_size
    end = min(len(data), (batch_idx + 1) * args.batch_size)

    # preprocessing should already discard empty documents
    for i in range(start, end):
        docid, head, body = data[i]
        inputs.append(body[:args.max_text_len])
        targets.append([word2idx[SOS]] + head + [word2idx[EOS]])

    # create a padded sequence
    input_lens = [len(x) for x in inputs]
    inputs = [pad_seq(x, max(input_lens)) for x in inputs]
    target_lens = [len(y) for y in targets]
    targets = [pad_seq(y, max(target_lens)) for y in targets]
    
    if not args.use_pointer_net:
        inputs = torch.LongTensor(inputs)
        targets = torch.LongTensor(targets)
    return inputs, targets, input_lens, target_lens

def mask_loss(logp, target_lens, targets):
    """
    logp: list of torch tensors, seq x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    logp = torch.stack(logp).transpose(0, 1) # after operation, b x s x d
    loss = 0
    for i in range(len(target_lens)):
        # the first one is SOS, so skip it
        idx = Variable(targets[i][1:target_lens[i]].view(-1, 1)) # s x 1
        logp_i = logp[i, :target_lens[i]-1, :] # s x d
        loss += torch.gather(logp_i, 1, idx).sum()
    # -: negative log likelihood
    return -loss

def another_mask_loss(logp_list, target_lens, targets):
    """
    logp_list: list of torch tensors, (seq - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    use_cuda = logp_list[0].is_cuda
    target_lens = target_lens.cuda() if use_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        logp = torch.gather(logp_list[i], 1, idx).view(-1)
        loss += logp[target_lens > i + 1].sum()
    return -loss

def build_local_index(inputs, targets):
    """
    inputs: list of index-text hybrid sequence for body (Eg: [92, EMP, 2, 78])
    targets: same as above, but for headline.
    """
    inps, tgts = [], []
    loc_word2idx, loc_idx2word = {}, {}
    loc_idx = args.vocab_size  # size of global index
    
    for inp, tgt in zip(inputs, targets):
        for i, word in enumerate(inp):
            if isinstance(word, str):
                if word not in loc_word2idx:   # an out-of-vocabulary word
                    loc_word2idx[word] = loc_idx
                    loc_idx2word[loc_idx] = word
                    loc_idx +=1
                inp[i] = loc_word2idx[word]
                
        for i, word in enumerate(tgt):
            # an out-of-vocabulary word that only exists in target will transform to UNK
            if isinstance(word, str):
                tgt[i] = loc_word2idx[word] if word in loc_word2idx else word2idx[UNK]
                
        inps.append(inp)
        tgts.append(tgt)
        
    inputs = torch.LongTensor(inps)
    targets = torch.LongTensor(tgts)
    return inputs, targets, loc_word2idx, loc_idx2word

def train(data):
    
    def data_transform(inputs, targets):
        loc_word2idx, loc_idx2word = {}, {}
        if args.use_pointer_net:
            inputs, targets, loc_word2idx, loc_idx2word = build_local_index(inputs, targets)
        if args.use_cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()
        return inputs, targets, loc_word2idx, loc_idx2word
        
    nbatch = len(data)
    ntest = nbatch // 50
    identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    random.seed(15213)
    random.shuffle(data)
    train_data = data[:-ntest]
    test_data = data[-ntest:]
    s2s = Seq2Seq.Seq2Seq(args)
    if args.use_cuda:
        s2s = s2s.cuda()
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr = args.learning_rate,
            weight_decay = args.l2)
    print(s2s)
    for param in s2s.parameters():
        print(param.data.size())
    print("identifier:", identifier)

    for ep in range(args.nepochs):
        ts = time.time()
        batch_idx = 0
        random.shuffle(train_data)
        epoch_loss, sum_len = 0, 0
        s2s.train(True)
        s2s.requires_grad = True
        for inputs, targets, input_lens, target_lens in train_data:
            # loc_word2idx, loc_idx2word: local oov indexing for a batch
            inputs, targets, loc_word2idx, loc_idx2word = data_transform(inputs, targets)
            oov_size = len(loc_word2idx)
            
            logp = s2s(inputs, input_lens, targets, oov_size)
            # loss = mask_loss(logp, target_lens, targets)
            loss = another_mask_loss(logp, target_lens, targets)
            sum_len += sum(target_lens)
            s2s_opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(s2s.parameters(), args.max_norm)
            s2s_opt.step()
            epoch_loss += loss.data[0]
            batch_idx += 1

            if batch_idx % 50 == 0:
                print("batch %d, time %.1f sec, loss %.2f" % (batch_idx,
                    time.time() - ts, loss.data[0]))
                s2s.train(False)
                s2s.requires_grad = False
                summarize(s2s, inputs, input_lens, targets, target_lens,
                        loc_idx2word, beam_search = False)
                sys.stdout.flush()
        train_loss = epoch_loss / sum_len

        s2s.train(False)
        s2s.requires_grad = False
        epoch_loss, sum_len = 0, 0
        for inputs, targets, input_lens, target_lens in test_data:
            inputs, targets, loc_word2idx, loc_idx2word = data_transform(inputs, targets)
            oov_size = len(loc_word2idx) 
            logp = s2s(inputs, input_lens, targets, oov_size)
            # loss = mask_loss(logp, target_lens, targets)
            loss = another_mask_loss(logp, target_lens, targets)
            sum_len += sum(target_lens)
            epoch_loss += loss.data[0]
        print("Epoch %d, train loss: %.2f, test loss: %.2f, #batch: %d, time %.2f sec"
                % (ep + 1, train_loss, epoch_loss / sum_len, batch_idx, time.time() - ts))
        model_fname = args.save_path + args.model_fpat % (identifier, ep + 1)
        torch.save(s2s, model_fname)

def idxes2sent(idxes, loc_idx2word):
    seq = []
    for idx in idxes:
        if idx2word[idx] == SOS:
            continue
        if idx2word[idx] == EOS:
            break
        seq.append(('%s_COPY' % loc_idx2word[idx]) if idx in loc_idx2word else idx2word[idx])
    # some characters may not be printable if not encode by utf-8
    return " ".join(seq).encode('utf-8').decode("utf-8")

def show_attn(input_text, output_text, gold_text, attn):
    """
    attn: output_s x input_s
    """
    input_words = [''] + input_text.split(' ')[:-1] # last one is empty
    output_words = [''] + output_text.split(' ')[:-1] + [EOS]
    attn = attn.data.cpu().numpy()[:len(output_words) - 1, :]

    fig = plt.figure()
    fig.set_size_inches(8, 5)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn, cmap='bone')
    fig.colorbar(cax, orientation='horizontal')

    # Set up axes
    ax.set_xticklabels(input_words, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("%s/figure/attn/%s.png" % (args.save_path, gold_text.replace(" ", "_").replace('/', '_')), dpi = 200)
    plt.close()

def summarize(s2s, inputs, input_lens, targets, target_lens, loc_idx2word, beam_search=True):
    logp, list_symbols, attns = s2s.summarize(inputs, input_lens, beam_search)
    list_symbols = torch.stack(list_symbols).transpose(0, 1) # b x s
    if beam_search is False and args.show_attn and args.attn_model != 'none':
        attns = torch.stack(attns).transpose(0, 1) # b x target_s x input_s

    for i in range(min(len(targets), 1)):
        """
        In train, we want to limit the summarize size for each batch to only 1
        to save time. In test, the data is prepared so that each batch only has
        1 instance.
        """
        symbols = list_symbols[i]
        decode_approach = 'Beam Search' if beam_search else 'Greedy Search'
        text = idxes2sent(inputs[i].cpu().numpy(), loc_idx2word)
        prediction = idxes2sent(symbols.cpu().data.numpy(), loc_idx2word)
        truth = idxes2sent(targets[i].cpu().numpy(), loc_idx2word)
        hyps[decode_approach].append(prediction)
        refs[decode_approach].append(truth)
        if beam_search is False and args.show_attn and args.attn_model != 'none':
            # only plot if it's not beam search
            show_attn(text, prediction, truth, attns[i, :, :])
        
        print("<Source Text>: %s" % text)
        print("<Ground Truth>: %s" % truth)
        print("<%s>: %s" % (decode_approach, prediction))
        print(80 * '-')

def test(model_path, testset, test_size=10000, is_text=True):
    
    def vectorize(raw_data):
        data_vec = []
        for data in raw_data:
            docid, headline, body = data
            headline = [word2idx[SOS]] + [word2idx.get(w, word2idx[UNK]) for w in headline] + [word2idx[EOS]]
            body = [word2idx.get(w, word2idx[UNK]) for w in body[:args.max_text_len]]
            data_vec.append((docid, headline, body))
        return data_vec
    
    s2s = torch.load(model_path, map_location=lambda storage, loc: storage)
    # switch to CPU for testing 
    s2s.use_cuda = False   
    s2s.train(False)
    s2s.requires_grad = False
    
    # transfrom into indice representation for testing 
    if is_text:
        testset = vectorize(testset)
    
    random.seed(15213)
    random.shuffle(testset)
    for _, headline, body in testset[:test_size]:
        inputs = torch.LongTensor([body])
        targets = torch.LongTensor([headline])
        summarize(s2s, inputs, [len(body)], targets, [len(headline)], beam_search=False)
        # summarize(s2s, inputs, [len(body)], targets, [len(headline)], beam_search=True)
        
    rouge = Rouge()
    for decode_approach in ["Greedy Search", "Beam Search"]:
        print("Decode Approach: {}".format(decode_approach))
        avg_score = rouge.get_scores(hyps[decode_approach], refs[decode_approach], avg=True)
        for metric, f1_prec_recl in avg_score.items():
            s = ', '.join(list(map(lambda x: '(%s, %.4f)' % (x[0], x[1]), f1_prec_recl.items())))
            print("%s: %s" % (metric, s))

def vec2text_from_full(test_size=500):
    idx2word_full = pickle.load(open(args.save_path + 'nyt/idx2word_full.pkl', 'rb'))
    data = pickle.load(open(args.save_path + 'nyt/nyt_eng_200912.pkl', 'rb'))[:test_size]
    data_text = []
    for docid, headline, body in data:
        if len(headline) > 0 and len(body) > 0:
            raw_headline = [idx2word_full[w] for w in headline]
            raw_body = [idx2word_full[w] for w in body]
            data_text.append((docid, raw_headline, raw_body))
    return data_text

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vecdata', type=str, default=
            # "/pylon5/ir3l68p/haomingc/1001Nights/standard_giga/train/train_data_std_v50000.pkl")
        "/pylon5/ir3l68p/haomingc/1001Nights/standard_giga/train/train_data_std_v50000_keepOOV.pkl")
    
    argparser.add_argument('--save_path', type=str, default=
            "/pylon5/ir3l68p/haomingc/1001Nights/")
    argparser.add_argument('--test_fpath', type=str, default=
            "/pylon5/ir3l68p/haomingc/1001Nights/standard_giga/test/test_data.pkl")
    argparser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    argparser.add_argument('--data_src', type=str, choices=['xml', 'std'], default='std')
    argparser.add_argument('--model_fpat', type=str, default="saved_model/s2s-s%s-e%02d.model")
    argparser.add_argument('--model_name', type=str, default="s2s-sO53Z-e22.model")
    argparser.add_argument('--use_cuda', action='store_true', default = False)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--emb_size', type=int, default=64)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--nepochs', type=int, default=30)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_text_len', type=int, default=32)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--teach_ratio', type=float, default=1)
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--attn_model', type=str, choices=['none', 'general', 'dot'], default='none')
    argparser.add_argument('--show_attn', action='store_true', default = False)
    # argparser.add_argument('--max_norm', type=float, default=100.0)
    argparser.add_argument('--l2', type=float, default=0.001)
    argparser.add_argument('--rnn_model', type=str, choices=['gru', 'lstm'], default='lstm')
    argparser.add_argument('--use_pointer_net', action='store_true', default = False)

    args = argparser.parse_args()
    for k, v in args.__dict__.items():
        print('- {} = {}'.format(k, v))
    # There are two types of data, vecdata already transforms the word to idx,
    # replace word OOV with idx of UNK; the other is raw text.
    vecdata = pickle.load(open(args.vecdata, "rb"))
    word2idx = vecdata["word2idx"]
    idx2word = vecdata["idx2word"]
    args.vocab_size = len(word2idx)
    
    # for evaluation 
    hyps, refs = {'Greedy Search':[], 'Beam Search':[]}, {'Greedy Search':[], 'Beam Search':[]}
    
    print("Running mode: {} model...".format(args.mode))
    if args.mode == 'train':
        train(group_data(vecdata["text_vecs"]))
    elif args.mode == 'test':
        model_path = args.save_path + args.model_name
        testset = None
        if args.data_src == 'xml':
            testset = vec2text_from_full() 
        if args.data_src == 'std':
            testset = pickle.load(open(args.test_fpath, 'rb'))
        test(model_path, testset)
