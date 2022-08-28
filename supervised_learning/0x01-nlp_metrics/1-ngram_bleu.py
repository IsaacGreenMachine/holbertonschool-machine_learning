#!/usr/bin/env python3
"""module for ngram_bleu function"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """

    bi_refs = []
    for ref in references:
        bi_refs.append(list2ngram(ref, n))
    sent = list2ngram(sentence, n)
    unigrams = len(sentence)
    bp = 1
    max = 0
    min_ref = min([len(ref) for ref in references])
    if unigrams <= min_ref:
        bp = np.exp(1-min_ref/len(sentence))
    sentence_count = []
    for group in bi_refs:
        word_count = {}
        for word in sent:
            word_count[word] = min(group.count(word), sent.count(word))
        sentence_count.append(word_count)
        if sum(word_count.values()) > max:
            max = sum(word_count.values())
            out = max / len(sent)
    return bp * out


def list2ngram(lst, n):
    bi_refs_in = []
    for i in range(len(lst) - (n - 1)):
        word = ""
        for j in range(n):
            word += lst[i + j]
            if j < (n - 1):
                word += " "
        bi_refs_in.append(word)
    return bi_refs_in
