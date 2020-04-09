from typing import Iterable, Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


Input = Tensor  # Tensor of word indexes of shape `(number_of_samples, sentence_length)`
Y = Tensor  # 1-D Tensor of label indexes


class FastText(nn.Module):
    def __init__(self, dict_size: int, dict_dim: int, n_labels: int, padding_idx=None):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(dict_size, dict_dim, padding_idx=padding_idx, sparse=True)
        self.linear = nn.Linear(dict_dim, n_labels)

    def forward(self, input: Input) -> Tensor:
        """
        :return: Tensor of shape `(number_of_samples, n_labels)
        """
        x = self.embedding(input)
        x = x.mean(dim=1)
        x = self.linear(x)
        return x


def train(model: FastText, batches: Iterable[Tuple[Input, Y]]):
    optimizer = torch.optim.SGD(model.parameters(recurse=True), lr=0.2)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(1, 20):
        loss_sum = 0
        no_batches = 0
        for x, y in batches:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            no_batches += 1
        print(f'epoch loss: {no_batches and (loss_sum / no_batches)}')


def predict_prob(model: FastText, input: Input) -> Tensor:
    with torch.no_grad():
        x = model(input)
        return F.softmax(x, dim=1)
