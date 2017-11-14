import pickle
import torch
from tokens import *
from torch.autograd import Variable

def pad_seq(seq, max_length):
    seq += [0] * (max_length - len(seq))
    return seq

def load_data(data_path):
    word2idx, idx2word, pos2idx, idx2pos = pickle.load(
        open(data_path + "dict.pkl", "rb"))
    docs = pickle.load(open(data_path + "text_keyword.pkl", "rb"))
    return docs, idx2word, idx2pos

def group_data(docs, batch_size):
    docs = sorted(docs, key = lambda x: len(x[1][1]), reverse = True)
    nbatches = (len(docs) + batch_size - 1) // batch_size
    return [next_batch(i, batch_size, docs) for i in range(nbatches)]

def extend_roots(roots, eos_indices):
    ext_roots = []
    start_idx = -1
    for i, root in enumerate(roots):
        end_idx = eos_indices[i]
        ext_roots += [root] * (end_idx - start_idx)
        start_idx = end_idx
    return ext_roots

def next_batch(idx, batch_size, docs):
    """
    docs: list of (docid, input_text, target_text)
    input_text, target_text: eos_indices, text, pos, roots
    """
    start = idx * batch_size
    end = min(len(docs), start + batch_size)

    inputs, targets, targets_roots = [], [], []
    inputs_eos_indices, targets_eos_indices = [], []
    for i in range(start, end):
        docid, input_text, target_text = docs[i]
        input_eos_indices, input_text, _, _ = input_text
        target_eos_indices, target_text, _, target_roots = target_text
        inputs.append(input_text)
        targets.append(target_text)
        inputs_eos_indices.append(input_eos_indices)
        targets_eos_indices.append(target_eos_indices)
        # target_roots only has one key word per sentence,
        # the extended form has one key word per word
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

def idxes2sent(idxes, idx2word):
    seq = [idx2word[idx].replace(EOS, EOS + "\n") for idx in idxes]
    # some characters may not be printable if not encode by utf-8
    return " ".join(seq).encode('utf-8').decode("utf-8")

def mask_loss(logp_list, target_lens, targets):
    """
    logp_list: list of torch tensors, (seq_len - 1) x batch x vocab_size
    target_lens: list of target lens
    targets: batch x seq
    """
    seq_len = targets.size(1)
    target_lens = torch.LongTensor(target_lens)
    use_cuda = logp_list[0].is_cuda
    target_lens = target_lens.cuda() if use_cuda else target_lens
    loss = 0
    # offset 1 due to SOS
    for i in range(seq_len - 1):
        idx = Variable(targets[:, i + 1].contiguous().view(-1, 1)) # b x 1
        logp = torch.gather(logp_list[i], 1, idx).view(-1)
        loss += logp[target_lens > i + 1].sum()
    return -loss
