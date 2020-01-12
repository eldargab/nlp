from typing import List, Iterator, Any, Tuple
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as nnf


Tok = int
Sentence = List[Tok]
Group = Any


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


class Net(nn.Module):
    def __init__(self, n_labels: int, voc_size: int, d: int):
        super(Net, self).__init__()

        self.word_vectors = nn.Parameter(torch.Tensor(voc_size, d), requires_grad=False)
        nn.init.uniform_(self.word_vectors, -1.0/d, 1.0/d)

        self.hidden = nn.Linear(d, n_labels)
        with torch.no_grad():
            self.hidden.weight.zero_()
            self.hidden.bias.zero_()

    def forward(self, sentences: List[Sentence]):
        voc_size, d = self.word_vectors.shape

        sv = torch.empty(len(sentences), d)  # sentence vectors

        for i, s in enumerate(sentences):
            sv[i] = self.word_vectors[s].sum() / len(s)

        def on_sv_grad(grad):
            if self.word_vectors.grad is None:
                self.word_vectors.grad = torch.zeros_like(self.word_vectors)
            for i, s in enumerate(sentences):
                self.word_vectors.grad[s] = grad[i] / len(s)

        sv.requires_grad_(True)
        sv.register_hook(on_sv_grad)

        return self.hidden(sv)


class Model:
    def __init__(self, d: int = 100, epochs: int = 10, batch_size: int = 10):
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
            sentence = tokens.fit(txt)
            if sentence:
                y.append(group_idx(group))
                sentences.append(sentence)

        no_samples = len(y)

        log(f'Finished data import ({no_samples} training samples, {len(groups)} labels)')

        y = torch.tensor(y)
        net = Net(len(groups), tokens.voc_size(), self.d)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

        number_of_batches = max(1, (no_samples // self.batch_size) - 1)

        for epoch in range(self.epochs):
            batch_start = 0
            epoch_loss = 0.0

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

                epoch_loss += loss.item()

            log(f'Finished epoch {epoch + 1} out of {self.epochs}, loss={epoch_loss / number_of_batches}')

        self.tokens = tokens
        self.net = net
        self.groups = groups
        self.group_list = [l for l, _ in sorted(groups.items(), key=lambda li: li[1])]

    def prob(self, samples: Iterator[str]):
        x = self.net.forward([self.tokens(s) for s in samples])
        x = nnf.softmax(x, dim=1)
        return x

    def predict(self, samples: Iterator[str]):
        p = self.prob(samples)
        return [self.get_group(i) for i in p.argmax(dim=1).numpy()]

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
