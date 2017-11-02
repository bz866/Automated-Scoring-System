#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:30:04 2017

@author: EricTseng
"""
import numpy as np
import pandas as pd
import nltk
from collections import Counter
from nltk import word_tokenize, pos_tag
import string
import enchant
from enchant.checker import SpellChecker
import re


def _cleanText(t):
    '''
    t string, raw text input
    ret t string, a list of words
    '''
    t = t.lower()
    t = re.sub(r'[^\w\s]','',t)
    t = re.sub(r'\s*(\(\d)|(\))\s*', '', t)
    #t = t.split()
    return t

def _nltktag(text):
    """
    Using nltk.word_tokenize to tag words as 'NN', 'DT'
    for extracting noun, verb, adj
    """
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    return tagged_words

def _wordCount(text):
    """
    input: string 
    output: int -- Count of words
    """
    return sum(Counter(text.split()).values())

def _longWordCount(text):
    """
    input: string
    output: int -- Count of Long words
    
    """
    #Average word length without stop words is 5.6
    ##threshold = 6
    long_words = [word for word in text.split() if len(word)>6]
    return sum(Counter(long_words).values())

def _partOfSpeechCount(text):
    
    tagged_words = _nltktag(text)
    #Noun Count
    listnn = [w[0] for w in tagged_words if w[1] in ['NN', 'NNP', 'NNPS','NNS']]
    nnCount = sum(Counter(listnn).values())
    #Verb Count
    listvb = [w[0] for w in tagged_words if w[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
    verbCount = sum(Counter(listvb).values())
    #Adjective Count
    listadj = [w[0] for w in tagged_words if w[1] in ['JJ', 'JJR', 'JJS']]
    adjCount = sum(Counter(listadj).values())
    #Adverb Count
    listadvb = [w[0] for w in tagged_words if w[1] in ['RR', 'RBR', 'RBS']]
    advbCount = sum(Counter(listadvb).values())
    return nnCount, verbCount, adjCount, advbCount

def _commaCount(text):
    return text.count(',')

def _punctuationCount(text):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    return count(text,set(string.punctuation)) 

def _sentenceCount(text):
    return len(nltk.sent_tokenize(text))

def _wordLengthAvg(text):
    l = text.split()
    return sum(map(len, l))/float(len(l))



def _spellingError(text):
    """
    return: Count of misspelled words
    """
    my_dict = enchant.Dict("en_US")
    my_checker = SpellChecker(my_dict)
    my_checker.set_text(text)
    return len([error.word for error in my_checker])

def _lexicalDiversity(t):
    """
    t input seq, String
    ---------
    return float ratio
    """
    return len(set(t)) / len(t)

def _quotationMark(t):
    '''
    t string, raw input
    ret li, ceil of pairs of quatation contained in input text
    '''
    li = re.findall('"',t)
    n = len(li)
    n = int(np.ceil(n/2))
    return n
    
def _exclamationMarks(text):
    return text.count('!')

def _featureExtraction(text):
    """
    input: essay as a long string
    
    output:feature vector
    elements in output: 
    1. word count 
    2. long word count
    3. noun word count
    4. verb count
    5. comma count
    6. punctuation count
    7. sentence count
    8. adjective count
    9. adverb count
    10. lexical diversity
    11. quatation mark
    12. word length
    13. spelling error
    14*.bracket count
    15*.exclamation count
    16*. Foreign words count
    """
    wordCount = _wordCount(text)
    longWordCount = _longWordCount(text)
    nounCount, verbCount, adjCount, advbCount = _partOfSpeechCount(text)
    commaCount = _commaCount(text)
    puncCount = _punctuationCount(text)
    sentCount = _sentenceCount(text)
    lexDiv = _lexicalDiversity(text)
    quatMarkCount = _quotationMark(text)
    avgWordLen = _wordLengthAvg(text)
    spelErrorCount = _spellingError(text)
    #brcktCount = _br
    exclamationCount = _exclamationMarks(text)
    
    f = np.array([wordCount, longWordCount, nounCount, verbCount, commaCount, puncCount, sentCount, 
                 adjCount, advbCount, lexDiv, quatMarkCount, avgWordLen, spelErrorCount])
    
    f_res = _addedByStep(f)
    
    return f_res #feature vector

def _addedByStep(vec):
    """
    input: vec
    output: vector that added up at each element
    """
    return [sum(vec[0:i+1]) for i in range(0,len(vec))] 


    
    