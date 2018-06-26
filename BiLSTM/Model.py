import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.2, bidirectional = False, pre_emb=None):
        super(LSTM, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_size, num_layers, bias=False, dropout=dropout, bidirectional=bidirectional)
        self.decoder = nn.Linear(hidden_size, 1)
        self.decoder_bi = nn.Linear(hidden_size*2, 1)
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_weights(pre_emb)
        
    def init_weights(self, pretrained_embedding):
            initrange = 0.1
            if(pretrained_embedding is not None):
                pretrained_embedding = pretrained_embedding.astype(np.float32)
                pretrained_embedding = torch.from_numpy(pretrained_embedding)
                self.encoder.weight.data = pretrained_embedding
            else:
                self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.rnn(emb, hidden)
        output = torch.mean(output, 0)
        output = torch.squeeze(output)
        if self.bidirectional:
            decoded = self.decoder_bi(output)
        else:
            decoded = self.decoder(output)
        return decoded, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.bidirectional == True:
            return (Variable(weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_()))
        else:
            return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))            
