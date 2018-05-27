from torch import nn
from constants import *
import torch

class EncoderRNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        
    def forward(self, input_, hidden):
        output, hidden = self.gru(self.embedding(input_), hidden)
        return output, hidden

    # TODO: other inits
    def initHidden(self, batch_size):
        en_hidden = torch.zeros(1, batch_size, self.hidden_size)
        if CUDA:
            en_hidden = en_hidden.cuda()
        return en_hidden
    
class DecoderRNN_GRU(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN_GRU, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        # TODO use transpose of embedding
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.LogSoftmax()
        
    def forward(self, input_, hidden):
        emb = self.embedding(input_).unsqueeze(1)
        # NB: Removed relu
        res, hidden = self.gru(emb, hidden)
        output = self.sm(self.out(res[:,0]))
        return output, hidden
    
    def initInput(self,batch_size):
        decoder_input = torch.LongTensor([SOS_TOKEN]*batch_size)
        if CUDA:
            decoder_input = decoder_input.cuda()
        return decoder_input