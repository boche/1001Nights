import torch.nn as nn

class Seq2Seq(nn.module):
    def __init__(self, args):
        self.emb = nn.Embedding(vocab_size, emb_size)

    def __init__(self, vocab_size, hidden_size, nlayers, encoder_emb,
            teacher_forcing_ratio):
        super(Seq2Seq, self).__init__()

    def forward(self):
        pass
