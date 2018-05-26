
# coding: utf-8

# # Sequence to Sequence
# 
# ## Chat Bot Model

# ## Import & Configs

# In[1]:


import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import RMSprop,Adam
from jieba import cut
from p3self.lprint import lprint
from p3self.matchbox import Trainer
from multiprocessing import Pool
from collections import Counter
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# In[2]:


BS = 256# Batch size

VOCAB_SEQ_IN = 3000
VOCAB_SEQ_OUT = 3000

SOS_TOKEN = 0
EOS_TOKEN = 1

LR = 5e-3
HIDDEN_SIZE = 256

VERSION = "0.0.1"

CUDA = torch.cuda.is_available()


# ## Loading data

# In[3]:


def read_hj_line(x):
    return tuple(list(i[2:] for i in x.split("\n")))

def cut_tkless(x):
    return " ".join(list(str(x)))

def cutline(x):
    return " ".join(list(cut(x)))

def load_xiaowangji():
    file = open("/data/chat/Dialog_Corpus/xiaohuangji50w_nofenci.conv")
    f=file.read()[2:]
    conv_block = f.split("\nE\n")
    conv_block
    
    p=Pool(6)
    conv_list=p.map(read_hj_line,conv_block)
    q,a=zip(*conv_list)
    
    q_l = p.map(cutline,q)
    a_l = p.map(cutline,a)
    
    file.close()
    return q_l,a_l

def load_xwj_tk_less():
    file = open("/data/chat/Dialog_Corpus/xiaohuangji50w_nofenci.conv")
    f=file.read()[2:]
    conv_block = f.split("\nE\n")
    conv_block
    
    p=Pool(6)
    conv_list=p.map(read_hj_line,conv_block)
    q,a=zip(*conv_list)
    
    q_l = p.map(cut_tkless,q)
    a_l = p.map(cut_tkless,a)
    
    file.close()
    return q_l,a_l


# In[4]:


class s2s_data(Dataset):
    def __init__(self,load_io, vocab_in, vocab_out, seq_addr, build_seq=False,
                 build_vocab = False,):
        """
        vocab_in,vocab_out are csv file addresses
        """
        self.load_io=load_io
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        self.seq_addr = seq_addr
        
        print("[Loading the sequence data]")
        
        if build_seq:
            self.i,self.o = self.load_io()
            np.save(self.seq_addr,[self.i,self.o])
        else:
            [self.i,self.o] = np.load(self.seq_addr).tolist()
        print("[Sequence data loaded]")
            
        assert len(self.i)==len(self.o),"input seq length mush match output seq length"
        
        self.N = len(self.i)
        print("Length of sequence:\t",self.N)
        
        if build_vocab:
            self.vocab_i = self.build_vocab(self.i)
            self.vocab_o = self.build_vocab(self.o)
            
            self.vocab_i.to_csv(self.vocab_in)
            self.vocab_o.to_csv(self.vocab_out)
            
            self.print_vocab_info()
        else:
            self.vocab_i = pd.read_csv(self.vocab_in).fillna("")
            self.vocab_o = pd.read_csv(self.vocab_out).fillna("")
                  
            self.print_vocab_info()
        
        print("building mapping dicts")
        self.i_char2idx,self.i_idx2char = self.get_mapping(self.vocab_i)
        self.o_char2idx,self.o_idx2char = self.get_mapping(self.vocab_o)
        
    def __len__(self):
        return self.N
    
    def __getitem__(self,idx):
        return self.seq2idx(self.i[idx],self.i_char2idx),self.seq2idx(self.o[idx],self.o_char2idx)
    
    def get_full_token(self,list_of_tokens):
        """
        From a list of list of tokens, to a long list of tokens, duplicate tokens included
        """
        return (" ".join(list_of_tokens)).split(" ")
    
    def get_mapping(self,vocab_df):
        char2idx=dict(zip(vocab_df["token"],vocab_df["idx"]))
        idx2char=dict(zip(vocab_df["idx"],vocab_df["token"]))
        return char2idx,idx2char
    
    def seq2idx(self,x,mapdict):
        return np.vectorize(lambda i:mapdict[i])(x.split(" ")).tolist()
    
    def get_token_count_dict(self,full_token):
        """count the token to a list"""
        return Counter(full_token)
    
    def build_vocab(self,seq_list):
        ct_dict = self.get_token_count_dict(self.get_full_token(seq_list))
        ct_dict["SOS_TOKEN"] = 9e9
        ct_dict["EOS_TOKEN"] = 8e9
        tk,ct = list(ct_dict.keys()),list(ct_dict.values())
        
        token_df=pd.DataFrame({"token":tk,"count":ct}).sort_values(by="count",ascending=False)
        return token_df.reset_index().drop("index",axis=1).reset_index().rename(columns={"index":"idx"}).fillna("")
    
    def print_vocab_info(self):
        self.vocab_size_i = len(self.vocab_i)
        self.vocab_size_o = len(self.vocab_o)
        
        print("[in seq vocab address]: %s,\t%s total lines"%(self.vocab_in,self.vocab_size_i))
        print("[out seq vocab address]: %s,\t%s total lines"%(self.vocab_out,self.vocab_size_o))
            
        print("Input sequence vocab samples:")
        print(self.vocab_i.sample(5))
        print("Output sequence vocab samples:")
        print(self.vocab_o.sample(5))

# We have to self difine a collate function
# becuz we take the longest sequence lengnth with in a batch as the seq length for the entire batch
def pad_collate(batch):
    i,o = zip(*batch)
    i_arr = pad_sequences(i,padding="post",)
    o_arr = pad_sequences(o,padding="post",)
    return torch.LongTensor(i_arr), torch.LongTensor(o_arr)
    


# In[5]:


# dl = DataLoader(s2s_data(load_xiaowangji,
#                          "/data/dict/chat_vocab_in.csv",
#                          "/data/dict/chat_vocab_out.csv",
#                          "/data/chat/xhj_seq.npy",
#                          build_seq=False,
#                          build_vocab=False),
#                 batch_size=BS)

# dl_gen = iter(dl)


# In[6]:


ds = s2s_data(load_xwj_tk_less,
                         "/data/dict/chat_vocab_char_in.csv",
                         "/data/dict/chat_vocab_char_out.csv",
                         "/data/chat/xhj_seq_char.npy",
                         build_seq = False,
                         build_vocab = False,)
dl = DataLoader(ds,
                batch_size=BS,
                collate_fn=pad_collate)

dl_gen = iter(dl)


# ## Seq2Seq Model

# In[7]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
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


# In[8]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
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


# In[9]:


encoder = EncoderRNN(dl.dataset.vocab_size_i,HIDDEN_SIZE)
decoder = DecoderRNN(HIDDEN_SIZE,dl.dataset.vocab_size_o)
criterion = nn.NLLLoss()
if CUDA:
    encoder.cuda()
    decoder.cuda()
    criterion.cuda()


# In[10]:


print(encoder)
print(decoder)


# In[11]:


en_opt = RMSprop(encoder.parameters(), lr=LR)
de_opt = RMSprop(decoder.parameters(), lr=LR)


# In[12]:


def save_s2s():
    torch.save(encoder.state_dict(), "/data/weights/enc_%s.pkl"%(VERSION))
    torch.save(decoder.state_dict(), "/data/weights/dec_%s.pkl"%(VERSION))
    
def load_s2s(version):
    encoder.load_state_dict(torch.load("/data/weights/enc_%s.pkl"%(version)))
    decoder.load_state_dict(torch.load("/data/weights/dec_%s.pkl"%(version)))


# In[13]:


load_s2s(VERSION)


# In[14]:


def train_action(*args,**kwargs):
    s1,s2 = args[0]
    ite = kwargs["ite"]
    if CUDA:
        s1,s2 = s1.cuda(),s2.cuda()
        
    batch_size = s1.size()[0]
    target_length = s2.size()[1]
    
    en_opt.zero_grad()
    de_opt.zero_grad()
    
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_output, encoder_hidden = encoder(s1,encoder_hidden)
    
    decoder_hidden = encoder_hidden # encoder passing hidden state to decoder!
    
    decoder_input  = decoder.initInput(batch_size)
    
    loss = 0
    for seq_idx in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
        
        idx_target = s2[:,seq_idx]
        
        loss += criterion(decoder_output,idx_target)
        decoder_input = idx_target # teacher forcing
        
    loss.backward()
    
    en_opt.step()
    de_opt.step()
    
    if ite%5==4:
        save_s2s()
    
    return {
        "loss":loss.item(),
    }


# In[ ]:


trainer = Trainer(dataset=ds,batch_size=4,print_on=2)
trainer.train_data.collate_fn = pad_collate
trainer.action = train_action


# In[ ]:


trainer.train(10)

