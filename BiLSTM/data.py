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

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def tokenize(string):
    return string.split()


def get_data(dataset):
    """
    Load the dataset and shuffle
    """
    dataset = dataset.reset_index(drop=True)
    dataset_set = []
    dataset_dict = {}
    for i in range(dataset.shape[0]):
        dataset_dict['label'] = int(dataset.loc[i, 'final_score'])
        dataset_dict['text'] = dataset.loc[i, 'essay']
        dataset_set.append(dataset_dict.copy())
    random.seed(1)
    random.shuffle(dataset_set)
    return dataset_set


def build_dictionary(training_datasets):
    """
    Extract vocabulary and build dictionary.
    """  
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        for example in dataset:
            word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter if word_counter[word] > 0])
    vocabulary = list(vocabulary)
    vocabulary = [PADDING, UNKNOWN] + vocabulary
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))

    return word_indices, len(vocabulary)



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
            example['label'] = torch.LongTensor([example['label']])



            