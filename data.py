from collections import namedtuple
import numpy as np
import torch

class Batch(namedtuple("Batch", "word_feat word_tgt label_feat label_tgt")):
    def to(self, device):
        return Batch(*[d.to(device) if d is not None else None for d in self])

    def explode(self):
        for i in range(self.label_tgt.shape[0]):
            yield Batch(
                self.word_feat[:, i:i+1, ...],
                self.word_tgt[:, i:i+1],
                self.label_feat[i:i+1, :],
                self.label_tgt[i:i+1]
            )

N_BATCH = 64
N_WORDS = 16

class Dataset():
    def __init__(self):
        #data0 = np.load('data/austen.txt.npy')
        #data1 = np.load('data/shakespeare.txt.npy')
        #data0 = np.load('data/aba.txt.npy')
        #data1 = np.load('data/aca.txt.npy')
        self.data0 = np.load('data/sentiment/sentiment.0.txt.npy')[:1000]
        self.data1 = np.load('data/sentiment/sentiment.1.txt.npy')[:1000]
        self.vocab = {}
        with open('data/vocab.txt') as fh:
            for line in fh:
                parts = line.strip().split()
                try:
                    word, tok = parts
                except:
                    word = "???"
                    tok, = parts
                self.vocab[int(tok)] = word

    def get_batch(self):
        words = []
        labels = []
        for i in range(N_BATCH):
            label = np.random.randint(2)
            labels.append(label)
            data = self.data0 if label == 0 else self.data1
            offset = np.random.randint(len(data)-N_WORDS-1)
            seq = data[offset:offset+N_WORDS+1]
            words.append(seq)

        label_feat = np.zeros((N_BATCH, 2))
        label_tgt = np.zeros((N_BATCH,))
        word_feat = np.zeros((N_WORDS, N_BATCH, len(self.vocab)))
        word_tgt = np.zeros((N_WORDS, N_BATCH))
        for i in range(N_BATCH):
            label_feat[i, labels[i]] = 1
            label_tgt[i] = labels[i]
            for j in range(N_WORDS):
                word_feat[j, i, words[i][j]] = 1
                #word_feat[i, words[i][j]] += 1
                word_tgt[j, i] = words[i][j+1]

        label_feat = torch.FloatTensor(label_feat)
        label_tgt = torch.LongTensor(label_tgt)
        word_feat = torch.FloatTensor(word_feat)
        word_tgt = torch.LongTensor(word_tgt)

        return Batch(word_feat, word_tgt, label_feat, label_tgt)

    def make_batch(self, obs_seq):
        obs = obs_seq[-1]
        word_feat = torch.zeros((1, 1, len(self.vocab)))
        word_feat[0, 0, obs] = 1
        return Batch(word_feat, None, None, None)
