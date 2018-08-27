#!/usr/bin/env python3

from collections import namedtuple
import heapq
from queue import PriorityQueue, Empty
import torch
from torch import nn

N_WORDVEC = 64
N_HIDDEN = 512

Hypothesis = namedtuple("Hypothesis", "score pred states parent aux")

def unroll(queue):
    out = []
    while True:
        try:
            item = queue.get(False)
            out.append(item)
        except Empty:
            break
    return out

class BeamSearch(object):
    def __init__(self, beam_size, make_batch, device):
        self.beam_size = beam_size
        self.make_batch = make_batch
        self.device = device

    def decode(self, models, states, first_obs):
        beam = [Hypothesis(0., (first_obs,), tuple(states), None, ())]
        for t in range(16):
            def candidates():
                for hyp in beam:
                    batch = self.make_batch(hyp.pred).to(self.device)
                    outputs = [
                        model(state, batch) for model, state in zip(models, hyp.states)
                    ]
                    scores, next_states, _ = zip(*outputs)
                    score = sum(scores).squeeze(0)
                    assert len(score.shape) == 1
                    top_score, top_idx = score.topk(self.beam_size)
                    for s, i in zip(top_score, top_idx):
                        aux = (
                            hyp.aux + ((scores[0][0, i].item(), scores[1][0, i].item()),)
                            if len(models) == 2
                            else ()
                        )
                        next_hyp = Hypothesis(
                            hyp.score + s,
                            hyp.pred + (i.item(),),
                            next_states,
                            hyp,
                            aux
                        )
                        yield next_hyp
            beam = heapq.nlargest(self.beam_size, candidates())
        best = max(beam)
        return best.pred[1:], best.aux


    ##def decode(self, label_feat, label_tgt, word_feat, word_tgt, style=False, flip=False):
    ##    del label_tgt, word_tgt
    ##    emb = self._emb(word_feat)
    ##    if flip:
    ##        label_feat = 1 - label_feat
    ##    rep_content = self._enc(emb.sum(dim=0))
    ##    rep_style = self._proj_style(self._emb_style(label_feat))
    ##    rep = rep_content + rep_style
    ##    last = word_feat[0, :, :]

    ##    out = [[] for _ in range(N_BATCH)]

    ##    hidden = rep.unsqueeze(0)
    ##    state = (hidden, torch.zeros_like(hidden))
    ##    for _ in range(N_WORDS):
    ##        emb = self._emb(last).unsqueeze(0)
    ##        dec_content, next_state = self._dec_content(emb, state)
    ##        logits_content = self._pred_content(dec_content.squeeze(0))
    ##        _, chosen = logits_content.max(dim=1)
    ##        chosen = chosen.data.cpu().numpy()
    ##        for i in range(N_BATCH):
    ##            out[i].append(chosen[i])

    ##        last = torch.zeros_like(last)
    ##        for i in range(len(chosen)):
    ##            last[i, chosen[i]] = 1
    ##        state = next_state

    ##    return out

class Encoder(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self._emb_content = nn.Linear(len(dataset.vocab), N_WORDVEC)
        self._enc_content = nn.LSTM(
            input_size=N_WORDVEC,
            hidden_size=N_HIDDEN,
            num_layers=1
        )
        self._enc_style = nn.Linear(2, N_HIDDEN)

    def forward(self, batch):
        emb_content = self._emb_content(batch.word_feat)
        _, (enc_content, _) = self._enc_content(emb_content)
        enc_style = self._enc_style(batch.label_feat)
        enc = enc_content + enc_style
        state = (enc, torch.zeros_like(enc))
        return state


class Decoder(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self._emb = nn.Linear(len(dataset.vocab), N_WORDVEC)
        self._dec = nn.LSTM(
            input_size=N_WORDVEC,
            hidden_size=N_HIDDEN,
            num_layers=1
        )
        self._pred = nn.Linear(N_HIDDEN, len(dataset.vocab))
        self._loss = nn.CrossEntropyLoss()

    def forward(self, state, batch):
        n_words, n_batch, n_vocab = batch.word_feat.shape
        emb = self._emb(batch.word_feat)
        dec, final_state = self._dec(emb, state)
        pred = self._pred(dec).view(n_words * n_batch, n_vocab)
        if batch.word_tgt is not None:
            loss = self._loss(pred, batch.word_tgt.view(n_words * n_batch))
        else:
            loss = None
        return torch.nn.functional.log_softmax(pred, dim=1), final_state, loss

class Model(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.encoder = Encoder(dataset)
        self.decoder = Decoder(dataset)

    def forward(self, batch):
        state = self.encoder(batch)
        pred, final_state, loss = self.decoder(state, batch)
        return loss
