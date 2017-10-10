
# coding: utf-8

# In[12]:


#!/usr/bin/env python

import numpy as np

def _split_into_words(sentences):
    words = list()
    for sentence in sentences:
        for word in sentence.split(" "):
                words.append(word)
    return words

def _get_ngrams(n, words):
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngrams.add(tuple(words[i : i + n]))
    return ngrams

def lcs_len(string, sub):
    """
    Return the length of the longest common subsequence between two string lists
    :param string: list of str
    :param sub: list of str
    :returns: length (int)
    """
    table = lcs(string, sub)
    n, m = len(string), len(sub)
    return table[n, m]

def lcs(string, sub):
    """
    Use a dynamic programming algorithm to calculate the longest common subsequence between two string lists
    :param string: list of str
    :param sub: list of str
    :returns: a table of lcs length
    """
    m, n = len(string), len(sub)
    table = dict()
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif string[i - 1] == sub[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def _p_r_f(lcs, m, n):
    """
    Calculate precision, recall and F score
    P_LCS = lcs / m
    R_LCS = lcs / n
    F_LCS = ((1 + beta^2) * P_LCS * R_LCS) / (R_LCS + beta^2 * P_LCS)
    """ 
    if m == 0:
        p_lcs = 0.0
    else:
        p_lcs = lcs / m
    if n == 0:
        r_lcs = 0.0
    else:
        r_lcs = lcs / n
    beta = 1 # Calculate F1 score here
    numerator = (1 + beta**2) * p_lcs * r_lcs
    denominator = r_lcs + beta**2 * p_lcs + 1e-8
    if denominator != 0:
        f_lcs = numerator / denominator
    else:
        f_lcs = 0.0
    return p_lcs, r_lcs, f_lcs

def cal_L_score(candidate_sentences, reference_sentences):
    """
    Calculate ROUGE_L score of candidate sentences and reference sentences
    """
    if len(candidate_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Error: Empty collection.")
    candidate_words = _split_into_words(candidate_sentences)
    reference_words = _split_into_words(reference_sentences)
    m = len(candidate_words)
    n = len(reference_words)
    lcs = lcs_len(candidate_words, reference_words)
    return _p_r_f(lcs, m, n)

def cal_n_score(candidate_sentences, reference_sentences, n):
    """
    Calculate ROUGE_N score of candidate sentences and reference sentences using n-gram
    """
    if len(candidate_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Error: Empty collection.")
    candidate_words = _split_into_words(candidate_sentences)
    reference_words = _split_into_words(reference_sentences)
    candidate_ngrams = _get_ngrams(n, candidate_words)
    reference_ngrams = _get_ngrams(n, reference_words)
    m = len(candidate_ngrams)
    n = len(reference_ngrams)
    overlap_ngrams = candidate_ngrams.intersection(reference_ngrams)
    overlap_len = len(overlap_ngrams)
    return _p_r_f(overlap_len, m, n)

def rouge(hypotheses, references):
    """
    Calculate average rouge scores for a set of hypotheses and references 
    """
    rouge_L = [
        cal_L_score([hyp], [ref])
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_L_p, rouge_L_r, rouge_L_f = map(np.mean, zip(*rouge_L))
    
    rouge_1 = [
        cal_n_score([hyp], [ref], 1)
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_p, rouge_1_r, rouge_1_f = map(np.mean, zip(*rouge_1))
    
    rouge_2 = [
        cal_n_score([hyp], [ref], 2)
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_2_p, rouge_2_r, rouge_2_f = map(np.mean, zip(*rouge_2))
    
    return {
        "rogue_L": rouge_L,
        "rouge_L/p_score": rouge_L_p,
        "rouge_L/r_score": rouge_L_r,
        "rouge_L/f_score": rouge_L_f,
        "rogue_1": rouge_1,
        "rouge_1/p_score": rouge_1_p,
        "rouge_1/r_score": rouge_1_r,
        "rouge_1/f_score": rouge_1_f,
        "rogue_2": rouge_2,
        "rouge_2/p_score": rouge_2_p,
        "rouge_2/r_score": rouge_2_r,
        "rouge_2/f_score": rouge_2_f,
    }

print (rouge(['today is a good day.', 'best day in my life'], ['good morning', 'good day in my life']))

