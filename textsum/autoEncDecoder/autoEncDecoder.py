import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import pickle 
import random

# from main import * 



""" Debug by printing the passed arguments for the model """
def debug_args(flags):
    logging.info('Parameters received:')
    for arg in vars(flags):
        value = getattr(flags, arg)
        logging.info('* {} = {}'.format(arg, value))



class autoEncDecoder(object):
    
    def __init__(self):
        self.embedding = None
        self.vocab_idx = None
        self.word_cnt = None
        self.params = None
        
        
    def load_embed_vocab(self):
        emb_dir = "/data/ASR5/haomingc/1001Nights/emb2010.pkl"
        word_cnt_dir = "/data/ASR5/haomingc/1001Nights/vocab2010.pkl"
        
        self.embedding, self.vocab_idx = pickle.load(open(embed_dir, 'rb'))
        print(type(self.embedding), self.embedding.shape)
        
        self.word_cnt = pickle.load(open(word_cnt_dir, 'rb'))
        print(type(self.word_cnt), len(self.word_cnt))



class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=500, n_layers=1, embedding_dim=300):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers)
        
    '''
        @param input: list of indices
        @param hidden: previous hidden state
    '''
    def forward(self, input, hidden_state):
        embedded = self.embedding(input).view(1, 1, -1)
        output = None
        
        for i in xrange(self.n_layers):
            output, hidden_state = self.gru(output, hidden_state)
        return output, hidden_state
    
    def init_hidden(self, use_cuda=False):
        ret = Variable(torch.zeros(1, 1, self.hidden_dim))
        print(ret)
        return ret.cuda() if use_cuda else ret


class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=200, n_layers=1, embedding_dim=300):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.softmax = nn.Softmax()
        
    
    def forward(self, input, hidden_state):
        output = self.embedding(input).view(1, 1, -1)
        for i in xrange(self.n_layers):
            output = F.relu(output)
            output, hidden_state = self.gru(output, hidden_state)
        
        out_ln = self.linear(output[0])
        out_sm = self.softmax(out_ln)
        return out_sm, hidden_state
    
    def init_hidden(self):
        ret = Variable(torch.zeros(1, 1, self.hidden_dim))
        print(ret)
        return ret.cuda() if use_cuda else ret



def main(argv):
    sys.argv = argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type = str, default = 'log', help = 'log dir')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help = 'mode: train or test')
    parser.add_argument('--input', type = str, help = 'input directory')
    parser.add_argument('--ouput', type = str, help = 'output directory')
    parser.add_argument('--epochs', type = int, default = 100, help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 100, help = 'batch size')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('--saved_model', type = str, default = None, help = 'directory of saved model')
    parser.add_argument('--cuda', action = 'store_true')

    args, _ = parser.parse_known_args()
    logging.basicConfig(level = 'DEBUG', format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_args(args)


if __name__ == '__main__':
#     main(sys.argv)

#     logging.basicConfig(level='DEBUG', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     model = autoEncDecoder()
#     model.load_embed_vocab()
    
    encoder = Encoder(300)
    encoder.init_hidden()
    
   

