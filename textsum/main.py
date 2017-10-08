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
from model import EncoderRNN, DecoderRNN


def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

def read_batch(word2idx):
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
        # the first one is SOS, so should skip
        idx = Variable(torch.LongTensor(targets[i][1:target_lens[i]]).view(-1, 1))
        logp_i = logp[i, :target_lens[i]-1, :]
        loss +=  torch.gather(logp_i, 1, idx).sum()
    return -loss

def save_model(model, name):
    torch.save(model, args.save_dir + name)

def test():
    encoder = EncoderRNN.EncoderRNN(len(word2idx), args.emb_size,
            args.hidden_size, args.nlayers)
    decoder = DecoderRNN.DecoderRNN(len(word2idx), args.hidden_size,
            args.nlayers, encoder.emb, 0.5)
    print(encoder)
    print(decoder)
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

    for ep in range(args.nepochs):
        epoch_end = False
        epoch_loss = 0
        reader.reset()
        while not epoch_end:
            inputs, targets, input_lens, target_lens, epoch_end = read_batch(
                    word2idx)
            encoder_output, encoder_hidden = encoder(torch.LongTensor(inputs), input_lens)
            logp = decoder(torch.LongTensor(targets), encoder_hidden)
            loss = mask_loss(logp, target_lens, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            encoder_opt.step()
            decoder_opt.step()
            epoch_loss += loss.data[0]
            print(loss.data[0])
    torch.save(encoder, args.save_dir + "encoder.md")
    torch.save(decoder, args.save_dir + "decoder.md")

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
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--hidden_size', type=int, default=10)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--nepochs', type=int, default=2)
    argparser.add_argument('--emb_size', type=int, default=80)
    argparser.add_argument('--max_text_len', type=int, default=1000)
    argparser.add_argument('--learning_rate', type=float, default=0.001)

    args = argparser.parse_args()
    reader = MyReader(args.dataset + "*")
    docs = reader.gen_docs()
    if args.build_vocab:
        build_vocab()
    elif args.build_emb:
        build_emb()
    else:
        word2idx, idx2word = get_vocab_idx()
        test()
