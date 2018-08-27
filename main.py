#!/usr/bin/env python3

from model import BeamSearch, Model
from data import Dataset

import torch
from torch import optim
from torch.optim import lr_scheduler as opt_sched

DEVICE = torch.device('cuda')

dataset = Dataset()
model = Model(dataset).to(DEVICE)
inference = BeamSearch(10, dataset.make_batch, DEVICE)
opt = optim.Adam(model.parameters(), 3e-4)
sched = opt_sched.ReduceLROnPlateau(opt, factor=0.5, verbose=True)
i = 0
running_loss = 0
for i in range(100000):
    batch = dataset.get_batch().to(DEVICE)
    loss = model(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    i += 1
    running_loss += loss.item()
    if (i+1) % 100 == 0:
        print(i+1)
        print(running_loss)
        sched.step(running_loss)
        running_loss = 0

        dec_batch = next(batch.explode())
        enc = model.encoder(dec_batch)
        enc_flipped = model.encoder(
            dec_batch._replace(label_feat=1-dec_batch.label_feat)
        )

        first_obs = dec_batch.word_feat[0, :, :].argmax(dim=1).item()
        dec, da = inference.decode(
            [model.decoder],
            [enc],
            first_obs
        )
        dec_transfer, dta = inference.decode(
            [model.decoder, model.decoder],
            [enc, enc_flipped],
            first_obs
        )
        print('ref', ' '.join(dataset.vocab[d] for d in dec_batch.word_tgt[:, 0].data.cpu().numpy()))
        print('dec', ' '.join(dataset.vocab[d] for d in dec))
        print("dec_transfer")
        for j in range(len(dec_transfer)):
            print("%10s %2.3f %2.3f" % ((dataset.vocab[dec_transfer[j]],) + dta[j]))
        print()

# just train an autoencoder and intersect with target language model
# reduced domain representation: words randomly deleted or replaced with
# out-of-domain nearest neighbors
