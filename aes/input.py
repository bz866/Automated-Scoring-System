import pickle
import os
import glob
import re
from subprocess import Popen, PIPE
import sys

currdir = os.getcwd()
os.chdir('%s' % currdir)

tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..'
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks

def grab_data(path, dictionary):
    sentences = []
    os.chdir(path)
    
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        print words
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

def grab_sentence(sentence, dictionary):
    sentences = []
    sentences = tokenize(sentence)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

if __name__ == '__main__':
    path = sys.argv[1]

    with open('twitter.dict.pkl', 'rb') as pickle_file:
        content = pickle.load(pickle_file)
        sentences = []

    for ff in glob.glob(path):
            with open(ff, 'r') as f:
                sentences.append(f.readline().strip())

    print(sentences)
    s = grab_sentence(sentences, content)

    f = open('smp.pkl', 'wb')
    pickle.dump(s, f, -1)
    f.close()
    print('finsihed')
