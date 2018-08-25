#!/usr/bin/env python3

import numpy as np

ABA = {0: 'a', 1: 'b'}
ACA = {0: 'a', 1: 'c'}

out_aba = []
out_aca = []
for i in range(1000):
    seq = np.random.randint(2, size=1+np.random.randint(10))
    out_aba.append(' '.join(ABA[j] for j in seq) + ' STOP')
    out_aca.append(' '.join(ACA[j] for j in seq) + ' STOP')


for fname, data in [('aba.txt', out_aba), ('aca.txt', out_aca)]:
    with open(fname, 'w') as fh:
        for line in data:
            fh.write(line + '\n')
