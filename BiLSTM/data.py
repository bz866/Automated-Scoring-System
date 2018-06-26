#import unicodedata
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import unicodedata
import re
import random
import collections
import numpy as np
import nltk

PADDING = "<PAD>"
UNKNOWN = "<UNK>"


def tokenize(string):
    """
    Substitute @Location1, @Location3 to @Location
    """
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def get_data(dataset):
    """
    Load the dataset, conver to lower, and shuffle
    """
    dataset = dataset.reset_index(drop=True)
    dataset_set = []
    dataset_dict = {}
    for i in range(dataset.shape[0]):
        dataset_dict['scaled_score2'] = int(dataset.loc[i, 'scaled_score2'])
        dataset_dict['label'] = int(dataset.loc[i, 'final_score'])
        dataset_dict['essay_set'] = dataset.loc[i, 'essay_set']
        dataset_dict['text'] = dataset.loc[i, 'essay'].lower()
        dataset_set.append(dataset_dict.copy())
    random.seed(1)
    random.shuffle(dataset_set)
    return dataset_set



def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    #use Counter() to keep track of vocab
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
            
    # keep words with frequency larger than 2
    vocabulary = set([word for word in word_counter if word_counter[word] > 2])
    vocabulary = list(vocabulary)
    # add Unknown and Padding
    vocabulary = [PADDING, UNKNOWN] + vocabulary
    # create word to index/index to word dictionary
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    index_to_word = dict(zip(range(len(vocabulary)), vocabulary))

    return word_indices, index_to_word, len(vocabulary)




def sentences_to_padded_index_sequences(word_indices, datasets, max_seq_length):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        for j, example in enumerate(dataset):
            example['text_index_sequence'] = torch.zeros(max_seq_length)

            token_sequence = tokenize(example['text'])
            padding = max_seq_length - len(token_sequence)

            for i in range(max_seq_length):
                if i >= len(token_sequence):
                    index = word_indices[PADDING]
                    pass
                else:
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                example['text_index_sequence'][i] = index

            example['text_index_sequence'] = example['text_index_sequence'].long().view(1,-1)
            example['label'] = torch.FloatTensor([example['label']])
            example['scaled_score2'] = torch.FloatTensor([example['scaled_score2']])




            
