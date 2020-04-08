from collections import OrderedDict
from typing import Callable, Iterator, TypeVar, Iterable

import numpy as np
import pandas as pd

T = TypeVar('T')


def frequencies(size_series: pd.Series):
    length = size_series.sum()
    size = size_series.sort_values(ascending=False)
    freq = size / length
    return pd.DataFrame.from_dict(OrderedDict([
        ('Size', size),
        ('Freq', freq),
        ('FreqCumSum', freq.cumsum()),
        ('SizeRevCumSum', size[::-1].cumsum()[::-1])
    ]))


def groups_summary(df: pd.DataFrame):
    sizes = df.groupby('Group').size()
    sizes = sizes[sizes > 0]
    return frequencies(sizes)


def group_size_by_week(df: pd.DataFrame, group: str):
    return df[df.Group == group].resample('W-MON', on='Date').size()


def describe_results(r: pd.DataFrame):
    hits = r.Group == r.Guess
    misses = ~hits

    stats = pd.DataFrame.from_dict(OrderedDict([
        ('Group', r.Group),
        ('Guess', r.Guess),
        ('Hit', hits),
        ('Miss', misses)
    ]))

    s = stats.groupby(['Group']).agg(
        Size=pd.NamedAgg('Hit', 'size'),
        Tp=pd.NamedAgg('Hit', 'sum')
    )

    fp = stats.groupby(['Guess']).agg(Fp=pd.NamedAgg('Miss', 'sum'))

    s = s.join(fp, how='outer')

    s['Precision'] = (s.Tp / (s.Tp + s.Fp)).map(lambda x: round(100 * x, ndigits=1))
    s['Recall'] = (s.Tp / s.Size).map(lambda x: round(100 * x, ndigits=1))
    s.sort_values(by='Size', ascending=False, inplace=True)

    s.fillna(0, inplace=True)
    s['Tp'] = s.Tp.astype(int)
    s['Fp'] = s.Fp.astype(int)

    return s


class IterableGenerator:
    def __init__(self, f: Callable[[], Iterator[T]]):
        self.f = f

    def __iter__(self) -> Iterator[T]:
        return self.f()


def iterable_generator(f:  Callable[[], Iterator[T]]) -> Iterable[T]:
    return IterableGenerator(f)


class ArrayBuilder:
    def __init__(self, dtype, initial_size=10, growth=1.25):
        self._growth = growth
        self._cap = initial_size
        self._size = 0
        self._data = np.empty(self._cap, dtype=dtype)

    def append(self, x):
        idx = self._size
        self._grow(self._size + 1)
        self._data[idx] = x

    def append_list(self, x):
        start = self._size
        end = self._size + len(x)
        self._grow(end)
        self._data[start:end] = x

    def _grow(self, size):
        self._size = size
        if size >= self._cap:
            new_cap = round(self._cap * self._growth)
            self._data.resize((new_cap,), refcheck=False)
            self._cap = new_cap

    def end(self):
        data = self._data
        data.resize((self._size,), refcheck=False)
        self._data = np.empty(0, dtype=data.dtype)
        self._size = 0
        self._cap = 0
        return data


class RaggedPaddedBatches:
    @staticmethod
    def from_list(data, batch_size, padding_value=0, dtype=None, size_dtype=np.int):
        batches_count = len(data) // batch_size
        sizes = np.empty(batches_count * batch_size, dtype=size_dtype)
        data_size = 0
        for bi in range(0, batches_count):
            start = bi * batch_size
            end = start + batch_size
            batch_item_size = max((len(data[i]) for i in range(start, end)), default=0)
            data_size += batch_item_size * batch_size
            sizes[start:end] = batch_size

        data_array = np.empty(data_size, dtype=dtype if dtype else np.array(data[0]).dtype)
        offset = 0
        for i in range(0, batches_count * batch_size):
            data_item = data[i]
            size = sizes[i]
            end = offset + size
            item_end = offset+len(data_item)
            data_array[offset:item_end] = data_item
            if end > item_end:
                data_array[item_end:end] = padding_value
            offset = end

        return RaggedPaddedBatches(data_array, sizes, batch_size)

    def __init__(self, data, sizes, batch_size):
        self._data = data
        self._sizes = sizes
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return len(self._sizes)

    @property
    def batches_count(self):
        return self.size // self.batch_size

    def shuffled_tensor_batches(self, dtype=None, y=None):
        import torch

        rng = np.random.default_rng(0)
        batches = rng.permutation(self.batches_count)
        tensor = torch.tensor(self._data, dtype=dtype)

        @iterable_generator
        def iterable():
            for batch in batches:
                start = batch * self.batch_size
                end = start + self.batch_size
                x = tensor[start:end].view(self.batch_size, -1)
                if y is None:
                    yield x
                else:
                    yield x, y[start:end]

        return iterable

    def tensor_batches(self, dtype=None):
        import torch

        tensor = torch.tensor(self._data, dtype=dtype)

        @iterable_generator
        def iterable():
            for batch in range(0, self.batches_count):
                start = batch * self.batch_size
                end = start + self.batch_size
                yield tensor[start:end].view(self.batch_size, -1)

        return iterable
