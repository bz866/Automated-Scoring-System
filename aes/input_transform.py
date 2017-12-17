import pickle
import os
import glob
import re
from subprocess import Popen, PIPE

currdir = os.getcwd()
os.chdir('%s' % currdir)

tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print ('Tokenizing..')
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    text = text.encode()
    tok_text, _ = tokenizer.communicate(text)
    #print(type(tok_text))
    #print(tok_text)
    #tok_text = tok_text.encode()
    toks = tok_text.decode().split('\n')[:-1]
    print ('Done')

    return toks

def grab_sentence(sentence, dictionary):
    sentences = []
    sentences = tokenize(sentence)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def tokenByDict(txt_path):
    with open('twitter.dict.pkl', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        
    sentences = []
    for ff in glob.glob(txt_path):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
            
    s = grab_sentence(sentences, content)
    return s

def seqByDict(sentence):
    with open('twitter.dict.pkl', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    s = grab_sentence(sentence, content)
    return s
    
