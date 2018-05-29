import torch

BS = 8# Batch size

VOCAB_SEQ_IN = 3000
VOCAB_SEQ_OUT = 3000

SOS_TOKEN = 0
EOS_TOKEN = 1

LR = 5e-3
HIDDEN_SIZE = 512
NB_LAYER = 2

VERSION = "0.0.3"
# "0.0.1" chars hidden =256
# "0.0.2" token hidden =512
# "0.0.3" layer=2 hidden =512

CUDA = torch.cuda.is_available()

CN_SEG = False

if CN_SEG:
    DICT_IN = "/data/dict/chat_vocab_in.csv"
    DICT_OUT = "/data/dict/chat_vocab_out.csv"
    SEQ_DIR = "/data/chat/xhj_seq.npy"
else:
    DICT_IN = "/data/dict/chat_vocab_in.csv"
    DICT_OUT = "/data/dict/chat_vocab_out.csv"
    SEQ_DIR = "/data/chat/xhj_seq_char.npy"
    
MAX_LEN = 20