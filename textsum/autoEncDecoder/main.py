import sys
import argparse
import numpy as np
import gensim
import time
import pickle
import torch
from MyReader import MyReader
from tokens import *
from model.EncoderRNN import EncoderRNN


def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq

def read_batch(word2idx):
    targets = []
    inputs = []
    while len(targets) < args.batch_size:
        # if at the end, return (None, None, None)
        docid, head, body = next(docs, (None, None, None))
        if docid is None:
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
    return inputs, targets, input_lens, target_lens

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

def test():
    encoder = EncoderRNN(len(word2idx), args.emb_size, args.hidden_size, 2)
    inputs, targets, input_lens, target_lens = read_batch(word2idx)
    output, hidden = encoder(torch.LongTensor(inputs), input_lens)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default=
            "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_2010*")
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
    argparser.add_argument('--emb_size', type=int, default=80)
    argparser.add_argument('--max_text_len', type=int, default=1000)

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
