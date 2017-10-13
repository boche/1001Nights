import sys
import argparse
import random
import string
import numpy as np
import gensim
import time
import pickle
from MyReader import MyReader
from tokens import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Seq2Seq

def pad_seq(seq, max_length):
    seq += [0] * (max_length - len(seq))
    return seq

def group_data(data):
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

    for i in range(start, end):
        docid, head, body = data[i]
        inputs.append(body[:args.max_text_len])
        targets.append([word2idx[SOS]] + head + [word2idx[EOS]])

    # create a padded sequence
    input_lens = [len(x) for x in inputs]
    inputs = [pad_seq(x, max(input_lens)) for x in inputs]
    target_lens = [len(y) for y in targets]
    targets = [pad_seq(y, max(target_lens)) for y in targets]
    return torch.LongTensor(inputs), torch.LongTensor(targets), input_lens, target_lens

def build_vocab():
    print("build vocab")
    vocab = {}
    docs = MyReader(args.rawdata).gen_docs()
    for docid, head, body in docs:
        print(docid, len(vocab))
        if len(head) > 0 and len(body) > 0:
            for w in head:
                vocab[w] = vocab.get(w, 0) + 1
            for w in body:
                vocab[w] = vocab.get(w, 0) + 1
    pickle.dump(vocab, open(args.save_path + args.vocab_fname, "wb"))

def build_emb():
    print("build emb")
    vocab_idx = {}
    vocab_cnt = pickle.load(open(args.save_path + args.vocab_fname, "rb"))
    pretrained_emb = gensim.models.KeyedVectors.load_word2vec_format(
            args.save_path + args.emb_bin_fname, binary=True)
    emb = []
    for word, cnt in vocab_cnt.items():
        if word in pretrained_emb:
            vocab_idx[word] = len(vocab_idx)
            emb.append(pretrained_emb[word])
    emb = np.asarray(emb)
    pickle.dump((emb, vocab_idx), open(args.save_path + args.emb_pkl_fname, "wb"))

def get_vocab_idx():
    word_cnt = pickle.load(open(args.save_path + args.vocab_fname, "rb"))
    word_cnt = sorted(word_cnt.items(), key= lambda x: x[1], reverse = True)
    word_cnt = word_cnt[:args.vocab_size]
    word2idx = {}
    idx2word = []
    for word in [SOS, EOS, UNK]:
        idx2word.append(word)
        word2idx[word] = len(word2idx)
    for word, _ in word_cnt:
        idx2word.append(word)
        word2idx[word] = len(word2idx)
    args.vocab_size = len(word2idx)
    return word2idx, idx2word

def mask_loss(logp, target_lens, targets):
    """
    logp: list of torch tensors, seq x batch x dim
    target_lens: list of target lens
    targets: batch x seq
    """
    logp = torch.stack(logp).transpose(0, 1) # b x s x d
    loss = 0
    for i in range(len(target_lens)):
        # the first one is SOS, so skip it
        idx = Variable(targets[i][1:target_lens[i]].view(-1, 1)) # s x 1
        logp_i = logp[i, :target_lens[i]-1, :] # s x d
        loss +=  torch.gather(logp_i, 1, idx).sum()
    # -: negative log likelihood
    return -loss

def train(data):
    nbatch = len(data)
    random.shuffle(data)
    ntest = nbatch // 10
    train_data = data[:-ntest]
    test_data = data[-ntest:]
    s2s = Seq2Seq.Seq2Seq(args)
    if args.use_cuda:
        s2s = s2s.cuda()
    print(s2s)
    for a in s2s.parameters():
        print(a.data.size())
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr=args.learning_rate,
            weight_decay = args.l2)
    identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    print("identifier:", identifier)

    for ep in range(args.nepochs):
        ts = time.time()
        batch_idx = 0
        random.shuffle(train_data)
        epoch_loss, sum_len = 0, 0
        for inputs, targets, input_lens, target_lens in train_data:
            if args.use_cuda:
                targets = targets.cuda()
                inputs = inputs.cuda()
            logp = s2s(inputs, input_lens, targets)
            loss = mask_loss(logp, target_lens, targets)
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
                summarize(s2s, inputs, input_lens, targets, target_lens)
                sys.stdout.flush()
        train_loss = epoch_loss / sum_len

        epoch_loss, sum_len = 0, 0
        for inputs, targets, input_lens, target_lens in test_data:
            if args.use_cuda:
                targets = targets.cuda()
                inputs = inputs.cuda()
            logp = s2s(inputs, input_lens, targets)
            loss = mask_loss(logp, target_lens, targets)
            sum_len += sum(target_lens)
            epoch_loss += loss.data[0]
        print("Epoch %d, train loss: %.2f, test loss: %.2f, #batch: %d, time %.2f sec"
                % (ep + 1, train_loss, epoch_loss / sum_len, batch_idx, time.time() - ts))
        epoch_loss = 0
        sum_len = 0
        # save model every epoch
        model_fname = args.save_path + args.model_fpat % (identifier, ep + 1)
        torch.save(s2s, model_fname)

def summarize(s2s, inputs, input_lens, targets, target_lens):
    logp, list_symbols = s2s.summarize(inputs, input_lens)
    list_symbols = torch.stack(list_symbols).transpose(0, 1) # b x s

    def idxes2sent(idxes):
        seq = ""
        for idx in idxes:
            seq += idx2word[idx] + " "
            if idx2word[idx] == EOS:
                break
        return seq.encode('utf-8')

    for i in range(min(len(targets), 3)):
        symbols = list_symbols[i]
        print("sample:", idxes2sent(symbols.cpu().data.numpy()))
        print("gt:", idxes2sent(targets[i].cpu().numpy()))
        print(80 * '-')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--rawdata', type=str, default=
            "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_20101*")
    # argparser.add_argument('--vecdata', type=str, default=
            # "/data/ASR5/haomingc/1001Nights/train_data_nyt_eng_2010_v50000.pkl")
    # argparser.add_argument('--save_path', type=str, default=
            # "/data/ASR5/bchen2/1001Nights/")
    argparser.add_argument('--vecdata', type=str, default=
            "/pylon5/ci560ip/bchen5/1001Nights/train_data_nyt_eng_2010_v50000.pkl")
    argparser.add_argument('--save_path', type=str, default=
            "/pylon5/ci560ip/bchen5/1001Nights/")
    argparser.add_argument('--emb_bin_fname', type=str, default=
            "GoogleNews-vectors-negative300.bin.gz")
    argparser.add_argument('--vocab_fname', type = str, default="vocab2010.pkl")
    argparser.add_argument('--emb_pkl_fname', type = str, default="emb2010.pkl")
    argparser.add_argument('--model_fpat', type = str, default="model/s2s-s%s-e%02d.model")

    argparser.add_argument('--build_vocab', action='store_true')
    argparser.add_argument('--build_emb', action='store_true')
    argparser.add_argument('--use_cuda', action='store_true', default = False)

    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--emb_size', type=int, default=128)
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--proj_size', type=int, default=128)
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--nlayers', type=int, default=1)
    argparser.add_argument('--nepochs', type=int, default=20)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_text_len', type=int, default=64)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--teach_ratio', type=float, default=0.5)
    # argparser.add_argument('--max_norm', type=float, default=100.0)
    argparser.add_argument('--l2', type=float, default=0.01)

    args = argparser.parse_args()
    for k, v in args.__dict__.items():
        print(k, v)
    if args.build_vocab:
        build_vocab()
    elif args.build_emb:
        build_emb()
    else:
        vecdata = pickle.load(open(args.vecdata, "rb"))
        word2idx = vecdata["word2idx"]
        idx2word = vecdata["idx2word"]
        args.vocab_size = len(word2idx)
        train(group_data(vecdata["text_vecs"]))
