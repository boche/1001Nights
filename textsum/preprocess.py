# coding: utf-8

import sys
import time
import pickle
from tokens import *
import gzip
import re
import glob
import random
import xml.etree.ElementTree as ET

class Extractor:
    def __init__(self, pattern):
        self.filenames = glob.glob(pattern)
        # print(self.filenames)
        self.file_id = 0
        self.doc_cnt = 0
        self.docs = []
        
    def textify(self, content):
        return [x.lower() for x in re.findall("([^\s)]+)\)", content)]
    
    def parse_doc(self, content):
        self.doc_cnt += 1
        if self.doc_cnt % 1000 == 0:
            print(' - Generating Doc {}...'.format(self.doc_cnt))
            
        root = ET.fromstring(content)
        docid = root.attrib['id']
        headline = root.find("HEADLINE")
        if headline is not None:
            headline = self.textify(headline.text)
        body = []
        text = root.find("TEXT")
        if text is not None:
            for para in text:
                body.extend(self.textify(para.text)) 
        self.docs.append((docid, headline, body))
    
    def gen_docs(self):
        print('Extracting documents...')
        content = ""
        for filename in self.filenames:
            print('Checking file: {}'.format(filename))
            f = gzip.open(filename, 'r')
            for line in f:    
                line = line.decode("utf-8")
                if "<DOC " in line:
                    content = ""
                content += line
                if "</DOC>" in line:
                    self.parse_doc(content)  
            f.close()
            self.file_id += 1
        print(' - Finish. Extracted {} documents'.format(self.doc_cnt))


def build_vocab(docs, path):
    print("Building vocabulary...")
    vocab = {}
    for idx, d in enumerate(docs):
        if idx % 1000 == 0:
            print(' - Scanning Doc {} out of {} docs...'.format(idx, len(docs)))
        docid, headline, body = d
        # print(docid, len(vocab))
        
        if len(headline) > 0 and len(body) > 0:
            for w in headline:
                vocab[w] = vocab.get(w, 0) + 1
            for w in body:
                vocab[w] = vocab.get(w, 0) + 1
    pickle.dump(vocab, open(path, "wb"))
    return vocab


def vectorize_docs(word2idx, docs):
    print("Vectoring training data...")
    docs_vec = []
    for idx, d in enumerate(docs):
        if idx % 1000 == 0:
            print(' - Vectoring Doc {} out of {} docs...'.format(idx, len(docs)))
            
        docid, headline, body = d
        if len(headline) > 0 and len(body) > 0:
            headline_vec = list(map(lambda x: word2idx.get(x, word2idx[UNK]), headline))
            body_vec = list(map(lambda x: word2idx.get(x, word2idx[UNK]), body))
            docs_vec.append((docid, headline_vec, body_vec))
    print(' - Finish.')
    return docs_vec


def build_index(vocab, vocab_size):
    word_cnt = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    word_cnt = word_cnt[:vocab_size]
    word2idx = {}
    idx2word = []
    for word in [SOS, EOS, UNK]:
        idx2word.append(word)
        word2idx[word] = len(word2idx)
    for word, _ in word_cnt:
        idx2word.append(word)
        word2idx[word] = len(word2idx)
    print(' - Finish.')
    return word2idx, idx2word


if __name__ == "__main__":
    # dataset = "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_2010*"
    dataset = "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_201002.xml.gz"
    
    version = dataset.split('/')[-1].replace('.xml.gz', '') 
    output_path = "/data/ASR5/haomingc/1001Nights/"
    vocab_pkl_path = '{}vocab_{}.pkl'.format(output_path, version)
    train_pkl_path = '{}train_data_{}.pkl'.format(output_path, version)
    
    reader = Extractor(dataset)
    reader.gen_docs()

    train_data = {}
    vocab_size = 50000
    vocab = build_vocab(reader.docs, vocab_pkl_path)
    
    word2idx, idx2word = build_index(vocab, vocab_size)
    train_data['word2idx'], train_data['idx2word'] = word2idx, idx2word
    
    train_data['text_vecs'] = vectorize_docs(word2idx, reader.docs)
    pickle.dump(train_data, open(train_pkl_path, "wb"))
    
    