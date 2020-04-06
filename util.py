from collections import OrderedDict
from typing import NamedTuple, Any

import numpy as np
import pandas as pd
import dask.dataframe as dd


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


class Ragged(NamedTuple):
    value: Any
    size: Any


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
        data.resize((self._size,), refcheck=True)
        self._data = np.empty(0, dtype=data.dtype)
        self._size = 0
        self._cap = 0
        return data

