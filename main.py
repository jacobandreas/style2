#!/usr/bin/env python3

import itertools as it
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler as opt_sched
from torch.autograd import Variable

data0 = np.load('data/austen.txt.npy')
data1 = np.load('data/shakespeare.txt.npy')
#data0 = np.load('data/aba.txt.npy')
#data1 = np.load('data/aca.txt.npy')
vocab = {}
with open('data/vocab.txt') as fh:
    for line in fh:
        word, tok = line.strip().split()
        vocab[int(tok)] = word

N_WORDVEC = 64
N_HIDDEN = 512
N_BATCH = 64
N_WORDS = 8

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._emb = nn.Linear(len(vocab), N_WORDVEC)
        #self._enc = nn.GRU(
        #    input_size=N_WORDVEC, hidden_size=N_HIDDEN, num_layers=1)
        self._enc = nn.Sequential(
            nn.Linear(N_WORDVEC, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN)
        )

        #self._emb_style = nn.Linear(2, N_HIDDEN)

        #self._proj_style = nn.Linear(N_HIDDEN, N_HIDDEN)
        #self._proj_content = nn.Linear(N_HIDDEN, N_HIDDEN)
        #self._pred_style = nn.Linear(N_HIDDEN, 2)
        #self._loss_style = nn.CrossEntropyLoss()

        self._dec_content = nn.LSTM(
            input_size=N_WORDVEC, hidden_size=N_HIDDEN, num_layers=1)
        self._pred_content = nn.Linear(N_HIDDEN, len(vocab))
        self._loss_content = nn.CrossEntropyLoss()

        #self._dec_joint = nn.GRU(
        #    input_size=N_WORDVEC, hidden_size=N_HIDDEN, num_layers=1)

    def forward(self, label_feat, label_tgt, word_feat, word_tgt):
        emb = self._emb(word_feat)
        rep_content = self._enc(emb.sum(dim=0))
        #_, enc = self._enc(emb)

        #rep = enc.squeeze(0)
        #rep_style = self._proj_style(rep)
        #rep_content = self._proj_content(rep)

        #rep_restyled = rep_content + self._emb_style(label_feat)

        #logits_style = self._pred_style(rep_style)
        #loss_style = self._loss_style(logits_style, label_tgt)

        hidden = rep_content.unsqueeze(0)
        state = (hidden, torch.zeros_like(hidden))
        dec_content, _ = self._dec_content(emb, state)
        logits_content = self._pred_content(dec_content).view(N_WORDS*N_BATCH, len(vocab))
        loss_content = self._loss_content(logits_content, word_tgt.view(N_WORDS*N_BATCH))

        #dec_full, _ = self._dec_content(emb, rep_restyled.unsqueeze(0))
        #logits_full = self._pred_content(dec_full).view(N_WORDS*N_BATCH, len(vocab))
        #loss_full = self._loss_content(logits_full, word_tgt.view(N_WORDS*N_BATCH))

        #return loss_content + loss_full - 0.001 * (1. / N_WORDS) * loss_style
        #return loss_content + loss_full, - 0.0001 * (1. / N_WORDS) * loss_style
        return loss_content

    def decode(self, label_feat, label_tgt, word_feat, word_tgt, style=False, flip=False):
        del label_tgt, word_tgt
        emb = self._emb(word_feat)
        #_, enc = self._enc(emb)
        #rep = self._proj_content(enc.squeeze(0))
        #if flip:
        #    label_feat = 1 - label_feat
        #if style:
        #    rep = rep + self._emb_style(label_feat)
        rep = self._enc(emb.sum(dim=0))
        last = word_feat[0, :, :]

        out = [[] for _ in range(N_BATCH)]

        hidden = rep.unsqueeze(0)
        state = (hidden, torch.zeros_like(hidden))
        for _ in range(N_WORDS):
            emb = self._emb(last).unsqueeze(0)
            dec_content, next_state = self._dec_content(emb, state)
            logits_content = self._pred_content(dec_content.squeeze(0))
            _, chosen = logits_content.max(dim=1)
            chosen = chosen.data.cpu().numpy()
            for i in range(N_BATCH):
                out[i].append(chosen[i])

            last = torch.zeros_like(last)
            for i in range(len(chosen)):
                last[i, chosen[i]] = 1
            state = next_state

        return out

def get_batch():
    words = []
    labels = []
    for i in range(N_BATCH):
        label = np.random.randint(2)
        labels.append(label)
        data = data0 if label == 0 else data1
        offset = np.random.randint(len(data)-N_WORDS-1)
        seq = data[offset:offset+N_WORDS+1]
        words.append(seq)

    label_feat = np.zeros((N_BATCH, 2))
    label_tgt = np.zeros((N_BATCH,))
    word_feat = np.zeros((N_WORDS, N_BATCH, len(vocab)))
    #word_feat = np.zeros((N_BATCH, len(vocab)))
    word_tgt = np.zeros((N_WORDS, N_BATCH))
    for i in range(N_BATCH):
        label_feat[i, labels[i]] = 1
        label_tgt[i] = labels[i]
        for j in range(N_WORDS):
            word_feat[j, i, words[i][j]] = 1
            #word_feat[i, words[i][j]] += 1
            word_tgt[j, i] = words[i][j+1]

    label_feat = Variable(torch.FloatTensor(label_feat))
    label_tgt = Variable(torch.LongTensor(label_tgt))
    word_feat = Variable(torch.FloatTensor(word_feat))
    word_tgt = Variable(torch.LongTensor(word_tgt))

    return label_feat, label_tgt, word_feat, word_tgt

model = Model()
opt = optim.Adam(model.parameters(), 5e-3)
sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True)
i = 0
running_loss = 0
while True:
    batch = get_batch()
    #loss_gen, loss_disc = model(*batch)
    loss_gen = model(*batch)
    opt.zero_grad()
    #loss_disc.backward(retain_graph=True)
    #nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    loss_gen.backward()
    opt.step()
    i += 1
    #loss_data = (loss_gen+loss_disc).data.cpu().numpy()[0]
    loss_data = loss_gen.item()
    running_loss += loss_data
    if (i+1) % 10 == 0:
        print(i+1)
        print(running_loss)
        sched.step(running_loss)
        running_loss = 0
        dec = model.decode(*batch)
        #dec_styled = model.decode(*batch, style=True)
        #dec_transfer = model.decode(*batch, style=True, flip=True)
        for j in range(3):
            print('ref', ' '.join(vocab[d] for d in batch[3][:, j].data.numpy()))
            print('uns', ' '.join(vocab[d] for d in dec[j]))
            #print('sty', ' '.join(vocab[d] for d in dec_styled[j]))
            #print('trs', ' '.join(vocab[d] for d in dec_transfer[j]))
            print()
