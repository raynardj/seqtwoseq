import torch

BS = 16# Batch size

VOCAB_SEQ_IN = 3000
VOCAB_SEQ_OUT = 3000

SOS_TOKEN = 0
EOS_TOKEN = 1

LR = 5e-3
HIDDEN_SIZE = 512

VERSION = "0.0.2"
# "0.0.1" chars
# "0.0.2" token

CUDA = torch.cuda.is_available()

CN_SEG = True

if CN_SEG:
    DICT_IN = "/data/dict/chat_vocab_in.csv"
    DICT_OUT = "/data/dict/chat_vocab_out.csv"
    SEQ_DIR = "/data/chat/xhj_seq.npy"
else:
    DICT_IN = "/data/dict/chat_vocab_in.csv"
    DICT_OUT = "/data/dict/chat_vocab_out.csv"
    SEQ_DIR = "/data/chat/xhj_seq_char.npy"