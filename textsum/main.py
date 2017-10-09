import sys
import argparse
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
    seq += [0 for i in range(max_length - len(seq))]
    return seq

def read_batch(word2idx, docs):
    targets = []
    inputs = []
    epoch_end = False
    while len(targets) < args.batch_size:
        # if at the end, return (None, None, None)
        docid, head, body = next(docs, (None, None, None))
        if docid is None:
            epoch_end = True
            break
        # only append non empty data
        if len(head) > 0 and len(body) > 0:
            head.insert(0, SOS)
            head.append(EOS)
            targets.append([word2idx[w] if w in word2idx else word2idx[UNK]
                for w in head])
            inputs.append([word2idx[w] if w in word2idx else word2idx[UNK]
                for w in body[:args.max_text_len]])

    if len(targets) == 0:
        # deal with the empty case separately because it will fail with
        # zip(*seq_pairs)
        return inputs, targets, [], [], epoch_end
    else:
        # sort by input length, create a padded sequence
        seq_pairs = sorted(zip(inputs, targets), key = lambda p: len(p[0]),
                reverse = True)
        inputs, targets = zip(*seq_pairs)
        input_lens = [len(x) for x in inputs]
        inputs = [pad_seq(x, max(input_lens)) for x in inputs]
        target_lens = [len(y) for y in targets]
        targets = [pad_seq(y, max(target_lens)) for y in targets]
        return inputs, targets, input_lens, target_lens, epoch_end

def build_vocab():
    print("build vocab")
    vocab = {}
    docs = MyReader(args.dataset + "*").gen_docs()
    for docid, head, body in docs:
        print(docid, len(vocab))
        if len(head) > 0 and len(body) > 0:
            for w in head:
                vocab[w] = vocab.get(w, 0) + 1
            for w in body:
                vocab[w] = vocab.get(w, 0) + 1
    pickle.dump(vocab, open(args.save_path + args.vocab_dir, "wb"))

def build_emb():
    print("build emb")
    vocab_idx = {}
    vocab_cnt = pickle.load(open(args.save_path + args.vocab_dir, "rb"))
    pretrained_emb = gensim.models.KeyedVectors.load_word2vec_format(
            args.save_path + args.emb_bin_dir, binary=True)
    emb = []
    for word, cnt in vocab_cnt.items():
        if word in pretrained_emb:
            vocab_idx[word] = len(vocab_idx)
            emb.append(pretrained_emb[word])
    emb = np.asarray(emb)
    pickle.dump((emb, vocab_idx), open(args.save_path + args.emb_pkl_dir, "wb"))

def get_vocab_idx():
    word_cnt = pickle.load(open(args.save_path + args.vocab_dir, "rb"))
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
        idx = Variable(targets[i][1:target_lens[i]].view(-1, 1))
        logp_i = logp[i, :target_lens[i]-1, :]
        loss +=  torch.gather(logp_i, 1, idx).sum()
    return -loss

def save_model(model, name):
    torch.save(model, args.save_dir + name)

def test():
    s2s = Seq2Seq.Seq2Seq(args).cuda()
    print(s2s)
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr=args.learning_rate)

    for ep in range(args.nepochs):
        epoch_end = False
        docs = MyReader(args.dataset + "*").gen_docs()
        batch_idx = 0
        epoch_loss = 0
        ts = time.time()
        while not epoch_end:
            batch_idx += 1
            inputs, targets, input_lens, target_lens, epoch_end = read_batch(
                    word2idx, docs)
            if len(targets) == 0:
                continue
            targets = torch.LongTensor(targets).cuda()
            inputs = torch.LongTensor(inputs).cuda()
            logp = s2s(inputs, input_lens, targets)
            loss = mask_loss(logp, target_lens, targets)
            s2s.zero_grad()
            loss.backward()
            s2s_opt.step()
            epoch_loss += loss.data[0]
            if batch_idx % 10 == 0:
                print("batch %d, size %d, time %.1f sec, loss %.2f" % (
                    batch_idx, len(targets), time.time() - ts, loss.data[0]))
                summarize(s2s, inputs, input_lens, targets, target_lens)
                sys.stdout.flush()
        print("Epoch %d, loss: %.2f, #batch: %d" % (ep + 1, epoch_loss, batch_idx))
    torch.save(s2s, args.save_dir + "ses.model")

def summarize(s2s, inputs, input_lens, targets, target_lens):
    logp, list_symbols = s2s.summarize(inputs, input_lens)
    list_symbols = torch.stack(list_symbols).transpose(0, 1) # b x s x d

    def idxes2sent(idxes):
        seq = ""
        for idx in idxes:
            seq += idx2word[idx] + " "
            if idx2word[idx] == EOS:
                break
        return seq

    for i in range(min(len(targets), 3)):
        symbols = list_symbols[i]
        print("sample:", idxes2sent(symbols.cpu().data.numpy()))
        print("gt:", idxes2sent(targets[i].cpu().numpy()))
        print(80 * '-')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default=
            "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_201012")
    argparser.add_argument('--save_path', type=str, default=
            "/data/ASR5/bchen2/1001Nights/")
    argparser.add_argument('--emb_bin_dir', type=str, default=
            "GoogleNews-vectors-negative300.bin.gz")
    argparser.add_argument('--vocab_dir', type = str, default="vocab2010.pkl")
    argparser.add_argument('--emb_pkl_dir', type = str, default="emb2010.pkl")
    argparser.add_argument('--build_vocab', action='store_true')
    argparser.add_argument('--build_emb', action='store_true')
    argparser.add_argument('--batch_size', type=int, default=40)
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--hidden_size', type=int, default=80)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--nepochs', type=int, default=20)
    argparser.add_argument('--emb_size', type=int, default=80)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_text_len', type=int, default=1000)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--teach_ratio', type=float, default=0.5)

    args = argparser.parse_args()
    print(args)
    if args.build_vocab:
        build_vocab()
    elif args.build_emb:
        build_emb()
    else:
        word2idx, idx2word = get_vocab_idx()
        test()
