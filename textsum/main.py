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
    return torch.LongTensor(inputs), torch.LongTensor(targets), input_lens, target_lens

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
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr=args.learning_rate,
            weight_decay = args.l2)
    identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    print(s2s)
    for param in s2s.parameters():
        print(param.data.size())
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
        print("sp:", idxes2sent(symbols.cpu().data.numpy()))
        print("gt:", idxes2sent(targets[i].cpu().numpy()))
        print(80 * '-')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--vecdata', type=str, default=
            # "/data/ASR5/haomingc/1001Nights/train_data_nyt_eng_2010_v50000.pkl")
    # argparser.add_argument('--save_path', type=str, default=
            # "/data/ASR5/bchen2/1001Nights/")
    argparser.add_argument('--vecdata', type=str, default=
            "/pylon5/ci560ip/bchen5/1001Nights/train_data_nyt_eng_2010_v50000.pkl")
    argparser.add_argument('--save_path', type=str, default=
            "/pylon5/ci560ip/bchen5/1001Nights/")
    argparser.add_argument('--model_fpat', type = str, default="model/s2s-s%s-e%02d.model")

    argparser.add_argument('--use_cuda', action='store_true', default = False)

    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--emb_size', type=int, default=128)
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--nlayers', type=int, default=3)
    argparser.add_argument('--nepochs', type=int, default=50)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_text_len', type=int, default=128)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--teach_ratio', type=float, default=0.5)
    # argparser.add_argument('--max_norm', type=float, default=100.0)
    argparser.add_argument('--l2', type=float, default=0.01)

    args = argparser.parse_args()
    for k, v in args.__dict__.items():
        print(k, v)
    vecdata = pickle.load(open(args.vecdata, "rb"))
    word2idx = vecdata["word2idx"]
    idx2word = vecdata["idx2word"]
    args.vocab_size = len(word2idx)
    train(group_data(vecdata["text_vecs"]))
