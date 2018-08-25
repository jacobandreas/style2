#!/usr/bin/env python3

from collections import Counter
import string
import numpy as np

FILE_NAMES = [
    'austen.txt', 'shakespeare.txt', 'dostoevsky.txt',
    'aba.txt', 'aca.txt'
]
KEEP_WORDS = 10000
INV = '*???*'
UNK = '*unk*'

def words(fname):
    with open(fname) as fh:
        for line in fh:
            clean = line.strip().lower().translate(
                str.maketrans('', '', string.punctuation))
            for word in clean.split():
                yield word

counter = Counter()
for fname in FILE_NAMES:
    for word in words(fname):
        counter[word] += 1

vocab = {}
vocab[INV] = 0
vocab[UNK] = 1
for word, count in counter.most_common(KEEP_WORDS):
    assert word not in vocab
    vocab[word] = len(vocab)

for fname in FILE_NAMES:
    data = []
    for word in words(fname):
        tok = vocab[word] if word in vocab else vocab[UNK]
        data.append(tok)
    np.save('%s.npy' % fname, np.asarray(data, dtype=np.int32))

with open('vocab.txt', 'w') as fh:
    for k, v in vocab.items():
        fh.write('%s %s\n' % (k, v))
