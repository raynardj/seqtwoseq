# seqtwoseq

Experiment on sequence to sequence, for chatbot

Support both GPU and CPU training.

### Files

The [model file](models.py) contains the encoder/decoder code

Train the model using [this notebook](seq2seq_chat.ipynb)

While training , you can have [this inference notebook](inference.ipynb) to check the model performance, loading the most recent weights the training notebook has saved.

[Constants file](constants.py) contains the configurations.

### Definitions

#### VERSION

Version between the model are controlled in following ways:
```python
VERSION = "0.0.3"
# "0.0.1" chars hidden =256
# "0.0.2" token hidden =512
# "0.0.3" layer=2 hidden =512
```
Please notice, even if you decide to move on the version number, don't delete it, comment it out and write some notes about it for further reference

The version number will be buried into model weights' file name.

#### CUDA

True or False, to the question: are we using the GPU

#### s2s_data

```class s2s_data``` is the dataset class (pytorch dataset class, the kind of class providing neat array data)。

It's universal to many datasets, all we have to do is to define a load function, pass as arg:load_io

In notebook, I defined 2 load functions, one for reading data as cn char, the other for tokenized cn words. The all return full_list_of_questions, full_list_of_answers.

* kwarg:build_seq 

The already built sequence(2 lists of sentences) can be saved to npy file. if kwarg:build_seq == False, we don't rebuild sequence again, just load the npy instead.

* kwarg:build_vocab

The already built/sorted vocabulary can bse saved to csv, if kwarg:build_vocab == False, we don't rebuild vocab again.

### Training

For detailed information, you'll find my seq2seq_chat.ipynb notebook very informative.



