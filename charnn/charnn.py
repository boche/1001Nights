import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import sys
import time
from model import CharNN

def load_file(filename):
  content = "" 
  charset = set()

  with open(filename) as f:
    for line in f:
      charset.update(set(line))
      content += line
  ch2ix = {ch: ix for ix, ch in enumerate(charset)}
  ix2ch = {ix: ch for ix, ch in enumerate(charset)}
  return content, ch2ix, ix2ch 

def sample():
  #TODO: replace with beam search
  ix = random.randint(0, input_size-1)
  line = ""
  line += ix2ch[ix]
  h = (Variable(torch.zeros(args.nlayers, 1, args.hidden_size).cuda()),
      Variable(torch.zeros(args.nlayers, 1, args.hidden_size).cuda()))
  for i in range(100):
    prob, h = model(Variable(torch.LongTensor([[ix]])).cuda(), h)
    prob = np.exp(prob.cpu().data.numpy().ravel())
    ix = np.random.choice(input_size, p = prob)
    # ix = int(np.argmax(prob))
    line += ix2ch[ix]
  return line

def random_sample():
  samples = torch.LongTensor(args.batch_size, args.chunk_len).cuda()
  targets = torch.LongTensor(args.batch_size, args.chunk_len).cuda()
  ndata = len(content)
  for i in range(args.batch_size):
    pos = random.randint(0, ndata - args.chunk_len - 1)
    samples[i,:] = torch.LongTensor(
        [ch2ix[ch] for ch in content[pos: pos + args.chunk_len]])
    targets[i,:] = torch.LongTensor(
        [ch2ix[ch] for ch in content[pos+1 : pos+args.chunk_len+1]])
  # print(samples)
  return Variable(samples), Variable(targets)

def train():
  nsamples_per_epoch = len(content) // (args.chunk_len * args.batch_size)
  # nsamples_per_epoch = 10
  criterion = nn.NLLLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  for e in range(args.nepochs):
    ts = time.time()
    epoch_loss = 0.0
    for i in range(nsamples_per_epoch):
      samples, targets = random_sample()
      h = None
      loss = 0
      for t in range(args.chunk_len):
        xout, h = model(samples[:, t], h)
        loss += criterion(xout, targets[:, t])
      model.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.data[0]
    print("Epoch %d, nll: %.4f, %.2f sec" % (e, epoch_loss, time.time() - ts))
    print("Sample: [%s]\n" % sample())
    sys.stdout.flush()

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('filename', type=str)
  argparser.add_argument('--nepochs', type=int, default=2000)
  # argparser.add_argument('--print_every', type=int, default=100)
  argparser.add_argument('--nlayers', type=int, default=2)
  argparser.add_argument('--learning_rate', type=float, default=0.001)
  argparser.add_argument('--chunk_len', type=int, default=200)
  argparser.add_argument('--batch_size', type=int, default=400)
  argparser.add_argument('--hidden_size', type=int, default=100)
  # argparser.add_argument('--cuda', action='store_true')
  args = argparser.parse_args()

  content, ch2ix, ix2ch = load_file(args.filename)
  input_size = len(ch2ix)
  model = CharNN(args.nlayers, input_size, args.hidden_size, args.batch_size)
  train()
