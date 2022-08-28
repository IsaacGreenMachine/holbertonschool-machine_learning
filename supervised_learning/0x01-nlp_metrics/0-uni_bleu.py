#!/usr/bin/env python3
"""module for uni_bleu function"""
import numpy as np


def uni_bleu(references, sentence):
    """
    that calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    unigrams = len(sentence)
    bp = 1
    max = 0
    min_ref = min([len(ref) for ref in references])
    if unigrams <= min_ref:
        bp = np.exp(1-min_ref/unigrams)

    # sentence_count is [{word:count}, {word:count}]
    # for each sentence in reference
    sentence_count = []
    for group in references:
        word_count = {}
        for word in sentence:
            word_count[word] = min(group.count(word), sentence.count(word))
        sentence_count.append(word_count)
        if sum(word_count.values()) > max:
            max = sum(word_count.values())
            out = max / unigrams

    print(bp, out)
    return bp * out
