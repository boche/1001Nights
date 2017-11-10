import sys
import argparse
import random
import string
import time
import pickle
import torch
from tokens import *
from preprocess import show_text
from model import Seq2Seq

def pad_seq(seq, max_length):
    seq += [0] * (max_length - len(seq))
    return seq

def load_data():
    word2idx, idx2word, pos2idx, idx2pos = pickle.load(
        open(args.data_path + "dict.pkl", "rb"))
    docs = pickle.load(open(args.data_path + "text_keyword.pkl", "rb"))
    return docs, idx2word, idx2pos

def group_data(docs):
    docs = sorted(docs, key = lambda x: len(x[1][1]), reverse = True)
    nbatches = (len(docs) + args.batch_size - 1) // args.batch_size
    a = next_batch(0, docs)
    return [next_batch(i, docs) for i in range(nbatches)]

def extend_roots(roots, eos_indices):
    ext_roots = []
    start_idx = -1
    for i in range(len(roots)):
        end_idx = eos_indices[i]
        ext_roots += [roots[i]] * (end_idx - start_idx)
        start_idx = end_idx
    return ext_roots

def next_batch(idx, docs):
    """
    docs: list of (docid, input_text, target_text)
    input_text, target_text: eos_indices, text, pos, roots
    """
    start = idx * args.batch_size
    end = min(len(docs), start + args.batch_size)

    inputs, targets = [], []
    inputs_eos_indices, targets_eos_indices = [], []
    targets_roots = []
    for i in range(start, end):
        docid, input_text, target_text = docs[i]
        input_eos_indices, input_text, _, _ = input_text
        target_eos_indices, target_text, _, target_roots = target_text
        inputs.append(input_text)
        targets.append(target_text)
        inputs_eos_indices.append(input_eos_indices)
        targets_eos_indices.append(target_eos_indices)
        targets_roots.append(extend_roots(target_roots, target_eos_indices))

    inputs_len = [len(x) for x in inputs]
    inputs = [pad_seq(x, max(inputs_len)) for x in inputs]
    targets_len = [len(y) for y in targets]
    targets = [pad_seq(y, max(targets_len)) for y in targets]
    targets_roots = [pad_seq(y, max(targets_len)) for y in targets_roots]
    return (torch.LongTensor(inputs), torch.LongTensor(targets),
            torch.LongTensor(inputs_eos_indices),
            torch.LongTensor(targets_eos_indices),
            torch.LongTensor(targets_roots), inputs_len, targets_len)

def train(data):
    random.shuffle(data)
    num_val = len(data) // 10
    train_data, val_data = data[:-num_val], data[-num_val:]
    s2s = Seq2Seq.Seq2Seq(args)
    s2s = s2s.cuda() if args.use_cuda else s2s
    s2s_opt = torch.optim.Adam(s2s.parameters(), lr = args.learning_rate,
            weight_decay = args.l2)

    for ep in range(args.nepochs):
        ts = time.time()
        random.shuffle(train_data)

        s2s.train(True)
        epoch_loss, sum_len, batch_idx = 0, 0, 0
        for batch in train_data:
            batch_idx += 1
            if args.use_cuda:
                batch = tuple(x.cuda() if isinstance(x, torch.LongTensor) else x
                        for x in batch)
            targets_len = batch[-1]
            loss = s2s(batch, False)
            s2s_opt.zero_grad()
            loss.backward()
            s2s_opt.step()

            epoch_loss += loss.data[0]
            # -len(targets_len) due to SOS
            sum_len += sum(targets_len) - len(targets_len)
        train_loss = epoch_loss / sum_len

        s2s.train(False)
        epoch_loss, sum_len = 0, 0
        for batch in test_data:
            if args.use_cuda:
                batch = tuple(x.cuda() if isinstance(x, torch.LongTensor) else x
                        for x in batch)
            targets_len = batch[-1]
            loss = s2s(batch, True)
            epoch_loss += loss.data[0]
            # -len(targets_len) due to SOS
            sum_len += sum(targets_len) - len(targets_len)
        print("Epoch %d, train loss %.2f, test loss %.2f, #batch %d, time %.2f sec"
                % (ep + 1, train_loss, epoch_loss / sum_len, batch_idx,
                    time.time() - ts))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default=
        "/pylon5/ci560ip/bchen5/1001Nights/keyword/")
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--emb_size', type=int, default=32)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--word_layers', type=int, default=2)
    argparser.add_argument('--sent_layers', type=int, default=2)
    argparser.add_argument('--max_text_len', type=int, default=100)
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--nepochs', type=int, default=30)
    argparser.add_argument('--max_sent_len', type=int, default=20)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    argparser.add_argument('--l2', type=float, default=0.001)
    argparser.add_argument('--use_cuda', action='store_true', default = False)
    argparser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    argparser.add_argument('--rnn_model', type=str, choices=['gru', 'lstm'], default='gru')

    args = argparser.parse_args()
    docs, idx2word, idx2pos = load_data()
    args.vocab_size = len(idx2word)

    data = group_data(docs[:2000])
    random.seed(15213)
    random.shuffle(data)
    num_test = len(data) // 10
    train_val_data, test_data = data[:-num_test], data[-num_test:]
    train(train_val_data)
