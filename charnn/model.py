import torch
import torch.nn as nn
from torch.autograd import Variable

class CharNN(nn.Module):
  def __init__(self, nlayers, input_size, hidden_size):
    super(CharNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.nlayers = nlayers
    # set batch size to 1 to use a common initial hidden state
    batch_size = 1

    self.embed = nn.Embedding(input_size, hidden_size)
    self.h0 = Variable(torch.randn(nlayers, batch_size, hidden_size),
        requires_grad = True)
    self.c0 = Variable(torch.randn(nlayers, batch_size, hidden_size),
        requires_grad = True)
    self.rnn = nn.LSTM(input_size = self.hidden_size,
        hidden_size = self.hidden_size,
        num_layers = self.nlayers)
    self.fc = nn.Linear(hidden_size, input_size)
    self.log_softmax = nn.LogSoftmax()

  def forward(self, xin, h):
    """
    xin: (batch_size, input_size)
    """
    xin = self.embed(xin)
    batch_size = xin.size(0)
    if h is None:
      h = (self.h0, self.c0)
    xout, h = self.rnn(xin.view(1, batch_size, -1), h)
    xout = self.fc(xout.view(batch_size, -1))
    xout = self.log_softmax(xout)
    return xout, h 
