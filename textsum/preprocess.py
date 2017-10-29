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

class FileReader:
    def __init__(self, files):
        self.filenames = files
        self.data = {}
        self.doc_cnt = 0
        self.docs = []
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = []
        self.i2w_full = None
    
    def textify(self, content):
        return [x.lower() for x in re.findall("([^\s)]+)\)", content)]
  
    def vectorize_raw(self, words):
        vec_words = []
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.word2idx)
            vec_words.append(self.word2idx[w])
        return vec_words
    
    def vectorize_compressed(self, words):
        vec_words = []
        for w in words:
            raw_w = self.i2w_full[w]
            vec_w = self.word2idx.get(raw_w, self.word2idx[UNK])
            vec_words.append(vec_w)
        return vec_words

    def build_index(self):
        logging.info("--- Indexing training data ---")
        word_cnt = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        word_cnt = word_cnt[:args.vocab_size]        
        top_words = list(map(lambda x: self.i2w_full[x[0]], word_cnt))
        
        for word in [SOS, EOS, UNK, NUM]:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)
        for word in top_words:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)
        self.data['word2idx'] = self.word2idx
        self.data['idx2word'] = self.idx2word
        
    def read_docs(self):
        logging.info("--- Reading documents from compressed files ---")
        for idx, filename in enumerate(self.filenames):
            logging.info("--- - Reading {} out of {} files ---".format(idx + 1, len(self.filenames)))
            docs = pickle.load(open(filename, "rb"))
            for _, doc in enumerate(docs):
                docid, headline, body = doc
                # discard document with empty headline or body
                if len(headline) > 0 and len(body) > 0:
                    for w in headline + body:
                        self.vocab[w] = self.vocab.get(w, 0) + 1
                    self.docs.append(doc)
    
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
    
    def gen_docs(self):
        if args.mode == 'compress':
            if args.source == 'xml':
                self.gen_compress_docs_from_xml()
            if args.source == 'standard':
                self.gen_compress_docs_from_std()

            i2w_path = '{}idx2word_full.pkl'.format(args.save_path)
            pickle.dump(self.idx2word, open(i2w_path, 'wb'))
            logging.info('--- Finish extracting {} documents ---'.format(self.doc_cnt))        
            logging.info('--- Extracted documents saved in {} ---'.format(args.save_path))
            logging.info('--- Format: list[tuple(docid, headline_vec, body_vec)] ---')
        
        if args.mode == 'extract':
            return self.gen_docs_from_compressed_file()

    def gen_compress_docs_from_std(self):
        
        def sp_token_transform(text):
            ret = []
            for w in text:
                if '#' in w:
                    w = NUM
                if w == '<unk>':
                    w = UNK
                ret.append(w)
            return ret 
             
        logging.info('--- Extracting training data from standard Gigawords ---')
        f_title = filter(lambda x: 'title' in x, self.filenames)[0]
        f_body = filter(lambda x: 'article' in x, self.filenames)[0]
        titles = open(f_title, 'r').readlines()
        bodies = open(f_body, 'r').readlines()   
        docs = []

        for docid, (title, body) in enumerate(zip(titles, bodies)):
            if docid % 1e5 == 0:
                logging.info('--- - {} out of {} docs ---'.format(docid, len(titles)))
            title, body = title.strip().split(), body.strip().split()
            title, body = sp_token_transform(title), sp_token_transform(body)
            title_vec = self.vectorize_raw(title)
            body_vec = self.vectorize_raw(body)
            docs.append((docid, title_vec, body_vec))
            
        f_name = args.save_path.split('/')[-2] + '_data'
        self.doc_cnt = len(docs)
        logging.info('--- Saving file: {} ---'.format(f_name))    
        pickle.dump(docs, open("{}.pkl".format(args.save_path + f_name), "wb"))
        
        return None

    def gen_compress_docs_from_xml(self):
        logging.info('--- Extracting documents by month ---')  
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
                        headline_vec = self.vectorize_raw(headline)
                        body_vec = self.vectorize_raw(body)
                        docs.append((docid, headline_vec, body_vec))
                        
                        # headline_rev = list(map(lambda x:self.idx2word[x], headline_vec))
                        # assert headline_rev == headline
            f.close()
            # save one month's text data into pickle 
            volume = filename.split('/')[-1].replace('.xml.gz', '')   
            logging.info('--- Saving file: {} ---'.format(volume))        
            pickle.dump(docs, open("{}.pkl".format(args.save_path + volume), "wb"))
        
        return None
    
    def gen_docs_from_compressed_file(self):
        self.read_docs()
        self.i2w_full = pickle.load(open(args.index_path, 'rb'))
        self.build_index()
        logging.info("--- Vectorizing training data ---")
        vec_data = []
        for idx, doc in enumerate(self.docs):
            if idx % 1e5 == 0:
                logging.info('--- - Vectoring {} out of {} docs ---'.format(idx, len(self.docs)))
            docid, headline, body = doc
            body_vec = self.vectorize_compressed(body)
            headline_vec = self.vectorize_compressed(headline)
            vec_data.append((docid, headline_vec, body_vec))
            
        logging.info("--- Training data ({} docs) generation complete ---".format(len(self.docs)))
        self.data['text_vecs'] = vec_data
        suffix = args.time_interval if args.source == 'xml' else 'std_all'
        vec_path = '{}vec_data_{}.pkl'.format(args.save_path, suffix)
        pickle.dump(self.data, open(vec_path, "wb"))
        logging.info('--- Vectorized documents saved in {} ---'.format(vec_path))
        logging.info('--- Format: dict{text_vecs, word2idx, idx2word} ---')
        return vec_data

def validate_time():
    start, end = list(map(lambda x: int(x), args.time_interval.split('-')))
    if end < start:
        raise Exception("ERROR: end time is earlier than start time.")
    for t in [start, end]:
        if not 0 < int(t % 100) < 13:
            raise Exception("ERROR: month format is invaild, should be YYYYMM")
    return start, end
        
        
def filter_files(pattern):
    candidates, files = glob.glob(pattern), None
    if args.mode == 'compress' or args.source == 'standard':
        files = candidates
    elif args.mode == 'extract':
        s, e = validate_time()
        files = list(filter(lambda x: s <= int(x[-10:-4]) <= e, candidates))
    logging.info('--- Found {} files ---'.format(len(files)))
    # print(files)
    return files


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    
    # # use data in self-defined format in '/nyt/'
    # argparser.add_argument('--raw_data', type=str, default=
    #         "/data/MM1/corpora/LDC2012T21/anno_eng_gigaword_5/data/xml/nyt_eng_20101*")
    # argparser.add_argument('--save_path', type=str, default=
    #         "/data/ASR5/haomingc/1001Nights/nyt/")
    # argparser.add_argument('--compressed_data', type=str, default=
    #     "/data/ASR5/haomingc/1001Nights/nyt/nyt_eng_*")
    # argparser.add_argument('--index_path', type=str, default=
    #         "/data/ASR5/haomingc/1001Nights/nyt/idx2word_full.pkl")
    
    # use generated data in '/standard_giga' for fair comparison among papers
    argparser.add_argument('--raw_data', type=str, default=
            "/data/ASR5/haomingc/1001Nights/standard_giga/train/*.txt")
    argparser.add_argument('--save_path', type=str, default=
            "/data/ASR5/haomingc/1001Nights/standard_giga/train/")
    argparser.add_argument('--compressed_data', type=str, default=
            "/data/ASR5/haomingc/1001Nights/standard_giga/train/*_data.pkl")
    argparser.add_argument('--index_path', type=str, default=
            "/data/ASR5/haomingc/1001Nights/standard_giga/train/idx2word_full.pkl")
    
    argparser.add_argument('--source', type=str, choices=['xml', 'standard'], default='standard')   
    argparser.add_argument('--mode', type=str, choices=['compress', 'extract'], help=
            "[compress] docs from raw XML or [extract] docs for training", default='compress')
    argparser.add_argument('--time_interval', type=str, default="201001-201012", help=
            "format: start_month-end_month, inclusive, range=[199407, 201012]")    
    argparser.add_argument('--vocab_size', type=int, default=50000)
    argparser.add_argument('--max_body_len', type=int, default=200)
    args = argparser.parse_args()
    logging.basicConfig(level = 'DEBUG', format= 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    pattern = None 
    if args.mode == 'compress':
        pattern = args.raw_data
    if args.mode == 'extract':
        pattern = args.compressed_data 
        
    files = filter_files(pattern)
    reader = FileReader(files)
    docs = reader.gen_docs()
    
    
    