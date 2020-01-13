from typing import List, Iterator, Any, Tuple
import re
import sys
import math
import numpy as np


Group = Any
Sentence = np.array  # array of word indexes


def split_words(txt: str):
    txt = txt.lower().replace('-', '_')
    return re.split(r'\W+', txt)


class Tokenizer:
    def __init__(self):
        self.voc = {}

    def fit(self, txt: str):
        return self._tokenize(txt, extend=True)

    def __call__(self, txt: str):
        return self._tokenize(txt, extend=False)

    def _tokenize(self, txt: str, extend: bool):
        out = []
        for w in split_words(txt):
            idx = self.voc.get(w)
            if idx is None:
                if extend:
                    idx = len(self.voc)
                    self.voc[w] = idx
                else:
                    continue
            out.append(idx)
        return out

    def voc_size(self):
        return len(self.voc)


def softmax(x: np.array):
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s


class Net:
    def __init__(self, n_labels: int, voc_size: int, d: int):
        self.word_vectors = np.random.default_rng().uniform(-1.0/d, 1.0/d, (voc_size, d))
        self.weights = np.zeros((n_labels, d))

    def forward(self, s: Sentence):
        sv = np.sum(self.word_vectors[s], axis=0) / len(s)
        o = np.matmul(self.weights, sv)
        return softmax(o)

    def backward(self, s: Sentence, y: int, lr):
        sv = np.sum(self.word_vectors[s], axis=0) / len(s)
        p = softmax(np.matmul(self.weights, sv))
        alpha = p * (-1.0 * lr)
        alpha[y] = (1 - p[y]) * lr
        self.word_vectors[s] += np.matmul(alpha, self.weights) / len(s)
        self.weights += np.outer(alpha, sv)
        return -math.log(p[y])


class Model:
    def __init__(self, d: int = 100, epochs: int = 5, batch_size: int = 10):
        self.d = d
        self.epochs = epochs
        self.batch_size = batch_size
        self.tokens = None
        self.net = None
        self.groups = None
        self.group_list = None

    def train(self, group_text_pairs: Iterator[Tuple[Group, str]], log=lambda s: print(s, file=sys.stderr)):
        groups = {}
        tokens = Tokenizer()

        def group_idx(l):
            idx = groups.get(l)
            if idx is None:
                idx = len(groups)
                groups[l] = idx
            return idx

        y = []
        sentences = []
        for group, txt in group_text_pairs:
            s = tokens.fit(txt)
            if s:
                y.append(group_idx(group))
                sentences.append(np.array(s))

        no_samples = len(y)
        log(f'Finished data import ({no_samples} training samples, {len(groups)} labels)')

        net = Net(len(groups), tokens.voc_size(), self.d)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(0, no_samples):
                epoch_loss += net.backward(sentences[i], y[i], 0.1)
            log(f'Finished epoch {epoch + 1} out of {self.epochs}, loss={epoch_loss / no_samples}')

        self.tokens = tokens
        self.net = net
        self.groups = groups
        self.group_list = [l for l, _ in sorted(groups.items(), key=lambda li: li[1])]

    def prob(self, txt: str):
        s = self.tokens(txt)
        return self.net.forward(s)

    def predict(self, txt: str):
        p = self.prob(txt)
        return self.get_group(np.argmax(p))

    def get_group(self, idx: int) -> Group:
        return self.group_list[idx]


def train(group_text_pairs: Iterator[Tuple[Group, str]]):
    m = Model()
    m.train(group_text_pairs)
    return m


def prepare_ft_file(group_text_pairs: Iterator[Tuple[Group, str]], out_file):
    with open(out_file, 'w') as out:
        for g, t in group_text_pairs:
            g = re.sub(r'\W', '_', g).lower()
            t = " ".join(split_words(t))
            print(f'__label__{g} {t}', file=out)
