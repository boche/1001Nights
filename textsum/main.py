import sys
import argparse
import numpy as np
import gensim
import time
import pickle
from MyReader import MyReader

def read_batch():
    data = []
    while len(data) < args.batch_size:
        # if at the end, return (None, None, None)
        docid, head, body = next(docs, (None, None, None))
        if docid is None:
            break
        # only append non empty data
        if len(head) > 0 and len(body) > 0:
            data.append((head, body))
    return data

def build_vocab():
    vocab = {}
    for docid, head, body in docs:
        print(docid, len(vocab))
        if len(head) > 0 and len(body) > 0:
            for w in head:
                w = w.lower()
                vocab[w] = vocab.get(w, 0) + 1
            for w in body:
                w = w.lower()
                vocab[w] = vocab.get(w, 0) + 1
    pickle.dump(vocab, open(args.save_path + args.vocab_dir, "wb"))

def build_emb():
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
    argparser.add_argument('--batch_size', type=int, default=100)
    
    args = argparser.parse_args()
    reader = MyReader(args.dataset + "*")
    docs = reader.gen_docs()
    if args.build_vocab:
        print("build vocab")
        build_vocab()
    elif args.build_emb:
        print("build emb")
        build_emb()
