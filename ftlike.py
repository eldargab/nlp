from functools import reduce
from typing import List, Iterator, Any, Tuple
from sklearn.utils import shuffle
import murmurhash
import math
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as nnf


Tok = int
Sentence = List[Tok]
Group = Any


def tokenize(txt: str, voc_size: int) -> Sentence:
    txt = txt.lower().replace('-', '_')
    words = re.split(r'\W+', txt)

    if voc_size.bit_length() > 30:
        raise ValueError('voc_size is too big')

    out = []
    added = set()

    def add_word(w):
        h = murmurhash.hash(w) % voc_size
        if h not in added:
            added.add(h)
            out.append(h)

    for idx, word in enumerate(words):
        add_word(word)
        if idx + 1 < len(words):
            add_word(word + ' ' + words[idx + 1])

    return out


class Net(nn.Module):
    def __init__(self, n_labels: int, voc_size: int, d: int):
        super(Net, self).__init__()

        self.projections = []
        for i in range(voc_size):
            v = nn.Parameter(torch.Tensor(d))
            with torch.no_grad():
                nn.init.uniform_(v, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))
            self.register_parameter(f'v{i}', v)
            self.projections.append(v)

        self.hidden = nn.Linear(d, n_labels)
        with torch.no_grad():
            nn.init.uniform_(self.hidden.weight, -1.0 / math.sqrt(d), 1.0 / math.sqrt(d))

    def forward(self, sentences: Iterator[Sentence]):
        zero_v = torch.zeros_like(self.projections[0])

        def embed(s: Sentence):
            assert len(s) > 0
            return reduce(lambda v, t: v + self.projections[t], s, zero_v) / len(s)

        x = torch.stack([embed(s if len(s) < 500 else s[0:500]) for s in sentences])
        x = self.hidden(x)
        return x


class Model:
    def __init__(self, voc_size: int, d: int = 100, epochs: int = 10, batch_size: int = 100):
        self.voc_size = voc_size
        self.d = d
        self.epochs = epochs
        self.net = None
        self.groups = None
        self._inv_groups = None
        self.batch_size = batch_size

    def train(self, group_text_pairs: Iterator[Tuple[Group, str]], log=lambda s: print(s, file=sys.stderr)):
        groups = {}

        def group_idx(l):
            idx = groups.get(l)
            if idx is None:
                idx = len(groups)
                groups[l] = idx
            return idx

        y = []
        sentences = []
        for group, txt in group_text_pairs:
            y.append(group_idx(group))
            sentences.append(tokenize(txt, self.voc_size))
        no_samples = len(y)

        log(f'Finished data import ({no_samples} training samples, {len(groups)} labels)')

        y, sentences = shuffle(y, sentences)

        y = torch.tensor(y)
        net = Net(len(groups), self.voc_size, self.d)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

        number_of_batches = max(1, (no_samples // self.batch_size) - 1)

        for epoch in range(self.epochs):
            batch_start = 0
            for batch_idx in range(number_of_batches):
                if batch_idx + 1 < number_of_batches:
                    batch_end = batch_start + self.batch_size
                else:
                    batch_end = no_samples

                optimizer.zero_grad()
                output = net(sentences[batch_start:batch_end])
                loss = criterion(output, y[batch_start:batch_end])
                loss.backward()
                optimizer.step()
                if batch_end == no_samples:
                    log(f'Finished epoch {epoch + 1} out of {self.epochs}, loss={loss.item()}')

        self.net = net
        self.groups = groups
        self._inv_groups = [l for l, _ in sorted(groups.items(), key=lambda li: li[1])]

    def prob(self, samples: Iterator[str]):
        x = self.net.forward(tokenize(s, self.voc_size) for s in samples)
        x = nnf.softmax(x, dim=1)
        return x

    def predict(self, samples: Iterator[str]):
        p = self.prob(samples)
        return [self.get_group(i) for i in p.argmax(dim=1).numpy()]

    def get_group(self, idx: int) -> Group:
        return self._inv_groups[idx]


def train(group_text_pairs: Iterator[Tuple[Group, str]]):
    m = Model(30000)
    m.train(group_text_pairs)
    return m
