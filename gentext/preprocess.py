import glob
import gzip
import numpy as np
import pickle
import sys
import time
import os
from tokens import *
import xml.etree.ElementTree as ET
from collections import Counter

def getidx(token, token2idx, idx2token):
    idx = token2idx.get(token, len(idx2token))
    if token not in token2idx:
        idx2token.append(token)
        token2idx[token] = idx
    return idx

def parse_doc(content):
    root = ET.fromstring(content)
    docid = root.attrib['id']
    sentences = root.find("sentences")
    if sentences is None:
        return None, None

    body = []
    for sentence in sentences:
        word_list, pos_list = [], []
        for token in sentence.find('tokens'):
            word = token.find('word').text
            pos = token.find('POS').text
            word2cnt[word] += 1
            word_list.append(getidx(word, word2idx, idx2word))
            pos_list.append(getidx(pos, pos2idx, idx2pos))
        root_id = 0
        for dep in sentence.find('basic-dependencies'):
            if dep.attrib['type'] == "root":
                root_id = int(dep.find('dependent').text) - 1 # -1 because 1 based
                break
        body.append((word_list, pos_list, word_list[root_id]))
    return docid, body

def index_file(data_path):
    filenames = glob.glob(data_path)
    dict_path = output_path + "rawdict.pkl"
    for filename in filenames:
        s = time.time()
        docs, content = [], ""
        for line in gzip.open(filename, 'r'):
            line = line.decode("utf-8")
            if "<DOC " in line:
                content = ""
            content += line
            if "</DOC>" in line:
                docid, body = parse_doc(content)
                if docid is not None:
                    docs.append((docid, body))
        basename = os.path.basename(filename)[:-7] + ".idx.pkl"
        print(basename, time.time() - s, len(word2idx), len(idx2word),
                len(pos2idx), len(idx2pos), len(docs))
        sys.stdout.flush()
        pickle.dump(docs, open(output_path + basename, "wb"))
        pickle.dump((idx2word, word2cnt, pos2idx, idx2pos),
                open(dict_path, "wb"))

def group_sentences(sentences, word2idx, pos2idx, raw_idx2word):
    sos_idx, eos_idx = word2idx.get(SOS), word2idx.get(EOS)
    unk_idx = word2idx.get(UNK)
    eos_indices, text, pos, roots = [], [], [], []
    for word_list, pos_list, root_idx in sentences:
        word_list = [sos_idx] + [word2idx.get(raw_idx2word[x], unk_idx)
                for x in word_list][:max_sent_len] + [eos_idx]
        eos_indices.append(len(word_list))
        text += word_list
        pos += [pos2idx[SOS]] + pos_list[:max_sent_len] + [pos2idx[EOS]]
        roots.append(word2idx.get(raw_idx2word[root_idx], unk_idx))
    eos_indices = list(np.cumsum(eos_indices) - 1)
    return eos_indices, text, pos, roots

def show_text(text, idx2word, idx2pos):
    eos_indices, text, pos, roots = text
    print([idx2word[x] for x in text])
    print([idx2pos[x] for x in pos])
    print([idx2word[x] for x in roots])

def convert_file():
    raw_idx2word, word2cnt, pos2idx, idx2pos = pickle.load(open(
        output_path + "rawdict.pkl", "rb"))
    top_words = [x[0] for x in word2cnt.most_common(vocab_size)]
    word2idx, idx2word = {}, []
    for word in [SOS, EOS, UNK] + top_words:
        idx2word.append(word)
        word2idx[word] = len(word2idx)
    for pos in [SOS, EOS]:
        idx2pos.append(pos)
        pos2idx[pos] = len(pos2idx)

    docs = []
    filenames = glob.glob(output_path + "*idx.pkl")
    for filename in filenames:
        s = time.time()
        for docid, body in pickle.load(open(filename, "rb")):
            if len(body) < max_input_sentences + max_target_sentences:
                continue
            input_text = group_sentences(body[:max_input_sentences], word2idx,
                    pos2idx, raw_idx2word)
            target_text = group_sentences(body[max_input_sentences:
                max_input_sentences+max_target_sentences], word2idx, pos2idx,
                raw_idx2word)
            docs.append((docid, input_text, target_text))
        print(filename, time.time() - s)
        sys.stdout.flush()
    pickle.dump(docs, open(output_path + "text_keyword.pkl", "wb"))
    pickle.dump((word2idx, idx2word, pos2idx, idx2pos),
            open(output_path + "dict.pkl", "wb"))

if __name__ == "__main__":
    input_path = "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/*gz"
    output_path = "/data/ASR5/bchen2/1001Nights/anno_eng_gigaword_5/"
    vocab_size = 50000
    max_input_sentences = 10
    max_target_sentences = 5
    max_sent_len = 20

    # word2idx, word2cnt, idx2word = {}, Counter(), []
    # pos2idx, idx2pos = {}, []
    # index_file(input_path)
    convert_file()
