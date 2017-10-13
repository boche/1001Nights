# coding: utf-8

import sys
import time
import pickle
import argparse
import logging
from tokens import *
import gzip
import re
import glob
import random
import xml.etree.ElementTree as ET

class Extractor:
    def __init__(self):
        self.filenames = None
        self.doc_cnt = 0
        self.word2idx = {}
        self.idx2word = []
    
    def textify(self, content):
        return [x.lower() for x in re.findall("([^\s)]+)\)", content)]
    
    def find_doc(self):
        self.filenames = glob.glob(args.raw_data)
        logging.info('--- Found {} files (months) ---'.format(len(self.filenames)))
    
    def parse_doc(self, content):
        if self.doc_cnt % 1000 == 0:
            logging.info('--- - Generating doc {} ---'.format(self.doc_cnt))
            
        root = ET.fromstring(content)
        docid = root.attrib['id']
        text = root.find("TEXT")
        headline = root.find("HEADLINE")
        if None in [headline, text]:
            return None, None, None
        
        self.doc_cnt += 1
        headline, body = self.textify(headline.text), []
        for para in text:
            body.extend(self.textify(para.text)) 
        return docid, headline, body
        
    def extend_word_index(self, words):
        vec_words = []
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.word2idx)
            vec_words.append(self.word2idx[w])
        return vec_words
    
    def gen_docs(self):
        logging.info('--- Extracting documents by month ---')  
        self.find_doc()
        
        for filename in self.filenames:
            docs, content = [], ""
            f = gzip.open(filename, 'r')
            for line in f:    
                line = line.decode("utf-8")
                if "<DOC " in line:
                    content = ""
                content += line
                if "</DOC>" in line:
                    docid, headline, body = self.parse_doc(content)
                    if docid is not None:
                        headline_vec = self.extend_word_index(headline)
                        body_vec = self.extend_word_index(body)
                        docs.append((docid, headline_vec, body_vec))
                        
                        # headline_rev = list(map(lambda x:self.idx2word[x], headline_vec))
                        # assert headline_rev == headline
            f.close()
            
            # save one month's text data into pickle 
            volume = filename.split('/')[-1].replace('.xml.gz', '')   
            logging.info('--- Saving file: {} ---'.format(volume))        
            pickle.dump(docs, open("{}.pkl".format(args.save_path + volume), "wb"))
        
        i2w_path = '{}idx2word_full.pkl'.format(args.save_path)
        pickle.dump(self.idx2word, open(i2w_path, 'wb'))
        logging.info('--- Finish extracting {} documents ---'.format(self.doc_cnt))        
        logging.info('--- Extracted documents saved in {} ---'.format(args.save_path))
        logging.info('--- Format: list[tuple(docid, headline_vec, body_vec)] ---')
        

class Vectorizer:
    def __init__(self):
        self.train_data = {}
        self.docs = []
        self.vocab = {}
        self.word2idx = {}  
        self.idx2word = []
        self.filenames = []
    
    def filter_docs(self):
        candidates, files = glob.glob(args.raw_data), []
        start, end = list(map(lambda x: int(x), args.time_interval.split('-')))
        if end < start:
            raise Exception("Error: end time is earlier than start time")
        if len(list(filter(lambda x: 0 < x < 13, [start % 100, end % 100]))) != 2:
            raise Exception("Error: month format is invaild")
        for candidate in candidates:
            month = int(candidate[-13:-7])
            if start <= month <= end:
                files.append(candidate)
        logging.info('--- Found {} files (months) ---'.format(len(files)))
        return files
        
    def read_docs(self):
        # TO DO
        return 
        
        
    def build_vocab(self):
        logging.info("--- Building vocabulary ---")
        for idx, d in enumerate(self.docs):
            if idx % 1000 == 0:
                logging.info("--- - Scanning {} out of {} docs ---".format(idx, len(self.docs)))
            docid, headline, body = d
            for w in headline + body:
                self.vocab[w] = self.vocab.get(w, 0) + 1
        self.train_data["vocab"] = self.vocab

    def build_index(self):
        logging.info("--- Indexing training data ---")
        word_cnt = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        word_cnt = word_cnt[:args.vocab_size]
        for word in [SOS, EOS, UNK]:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)
        for word, _ in word_cnt:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)
        self.train_data['word2idx'] = self.word2idx
        self.train_data['idx2word'] = self.idx2word
        
    def vec_docs(self):
        self.filter_docs()
        self.read_docs()
        self.build_vocab()
        self.build_index()

        logging.info("--- Vectorizing training data ---")
        vec_data = []
        for idx, d in enumerate(self.docs):
            if idx % 1000 == 0:
                print('--- - Vectoring {} out of {} docs ---'.format(idx, len(self.docs)))
            docid, headline, body = d
            body_vec = list(map(lambda x: self.word2idx.get(x, self.word2idx[UNK]), body))
            headline_vec = list(map(lambda x: self.word2idx.get(x, self.word2idx[UNK]), headline))
            vec_data.append((docid, headline_vec, body_vec))
            
        logging.info("--- Finish vectoring ---")
        self.train_data['text_vecs'] = vec_data
        vec_path = '{}vec_data_{}.pkl'.format(args.save_path, args.time_interval)
        pickle.dump(self.train_data, open(vec_path, "wb"))
        logging.info('--- Vectorized documents saved in {} ---'.format(vec_path))
        logging.info('--- Format: dict{text_vecs, vocab, word2idx, idx2word} ---')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--raw_data', type=str, default=
            "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_*")
    argparser.add_argument('--save_path', type=str, default=
            "/data/ASR5/haomingc/1001Nights/nyt/")
    argparser.add_argument('--index_path', type=str, default=
            "/data/ASR5/haomingc/1001Nights/nyt/idx2word_full.pkl")
    argparser.add_argument('--mode', type=str, choices=['extract', 'index'], help=
            "[extract] docs from raw XML or [index] docs for training", default='extract')
    argparser.add_argument('--time_interval', type=str, default="201001-201012", help=
            "format: start_month-end_month, inclusive, range=[199407, 201012]")
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--max_title_len', type=int, default=20)
    argparser.add_argument('--max_body_len', type=int, default=200)
    args = argparser.parse_args()
    logging.basicConfig(level = 'DEBUG', format= 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.mode == 'extract':
        reader = Extractor()
        reader.gen_docs()
    # if args.mode == 'vectorize':
    #     vectorizer = Vectorizer()
    #     vectorizer.vec_docs()
    
    