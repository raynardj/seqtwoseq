from torch import nn
from constants import *
import torch

# GRU

class EncoderRNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN_GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.cuda_ = CUDA
        
    def forward(self, input_, hidden):
        output, hidden = self.gru(self.embedding(input_), hidden)
        return output, hidden

    # TODO: other inits
    def initHidden(self, batch_size):
        en_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if self.cuda_:
            en_hidden = en_hidden.cuda()
        return en_hidden
    
class DecoderRNN_GRU(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN_GRU, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        # TODO use transpose of embedding
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=-1)
        self.cuda_ = CUDA
        
    def forward(self, input_, hidden):
        emb = self.embedding(input_).unsqueeze(1)
        # NB: Removed relu
        res, hidden = self.gru(emb, hidden)
        output = self.sm(self.out(res[:,0]))
        return output, hidden
    
    def initInput(self,batch_size):
        decoder_input = torch.LongTensor([SOS_TOKEN]*batch_size)
        if self.cuda_:
            decoder_input = decoder_input.cuda()
        return decoder_input
    
# LSTM

class EncoderRNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.cuda_ = CUDA
        
    def forward(self, input_, hidden):
        print(self.embedding(input_).size())
        output, hidden = self.lstm(self.embedding(input_), hidden)
        return output, hidden

    # TODO: other inits
    def initHidden(self, batch_size):
        en_hidden = torch.zeros(1, batch_size, self.hidden_size)
        if self.cuda_:
            en_hidden = en_hidden.cuda()
        return en_hidden
    
class DecoderRNN_LSTM(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        # TODO use transpose of embedding
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax(dim=-1)
        self.cuda_ = CUDA
        
    def forward(self, input_, hidden):
        emb = self.embedding(input_).unsqueeze(1)
        # NB: Removed relu
        res, hidden = self.lstm(emb, hidden)
        output = self.sm(self.out(res[:,0]))
        return output, hidden
    
    def initInput(self,batch_size):
        decoder_input = torch.LongTensor([SOS_TOKEN]*batch_size)
        if self.cuda_:
            decoder_input = decoder_input.cuda()
        return decoder_input