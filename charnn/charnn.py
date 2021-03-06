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
  ans = ""
  max_prob = 0.0

  for i in range(10):
    ix = random.randint(0, input_size-1)
    line = ""
    line += ix2ch[ix]
    h = None
    sum_prob = 0.0
    for i in range(100):
      prob, h = model(Variable(torch.LongTensor([[ix]])), h)
      prob = np.exp(prob.cpu().data.numpy().ravel())
      ix = np.random.choice(input_size, p = prob)
      sum_prob += prob[ix]
      line += ix2ch[ix]

    if sum_prob > max_prob:
      max_prob = sum_prob
      ans = line
  return ans

def random_sample():
  samples = torch.LongTensor(args.batch_size, args.chunk_len)
  targets = torch.LongTensor(args.batch_size, args.chunk_len)
  ndata = len(content)
  for i in range(args.batch_size):
    pos = random.randint(0, ndata - args.chunk_len - 1)
    idxes = [ch2ix[ch] for ch in content[pos: pos + args.chunk_len + 1]]
    samples[i,:] = torch.LongTensor(idxes[:-1])
    targets[i,:] = torch.LongTensor(idxes[1:])
  return Variable(samples), Variable(targets)

def train():
  criterion = nn.NLLLoss()
  nsamples_per_epoch = len(content) // (args.chunk_len * args.batch_size)
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
        loss += criterion(xout, targets[:, t].cuda())
      model.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.data[0]
    print("Epoch %d, nll: %.4f, %.2f sec" % (e, epoch_loss, time.time() - ts))
    if e % args.print_every == 0:
      print("Sample: [%s]\n" % sample())
    sys.stdout.flush()

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('filename', type=str)
  argparser.add_argument('--nepochs', type=int, default=2000)
  argparser.add_argument('--print_every', type=int, default=10)
  argparser.add_argument('--nlayers', type=int, default=2)
  argparser.add_argument('--learning_rate', type=float, default=0.001)
  argparser.add_argument('--chunk_len', type=int, default=200)
  argparser.add_argument('--batch_size', type=int, default=400)
  argparser.add_argument('--hidden_size', type=int, default=100)
  args = argparser.parse_args()

  content, ch2ix, ix2ch = load_file(args.filename)
  input_size = len(ch2ix)
  model = CharNN(args.nlayers, input_size, args.hidden_size, args.batch_size).cuda()
  print(model)
  train()
