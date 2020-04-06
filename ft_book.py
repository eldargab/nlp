#%%
from ipython_setup import aimport
from builder import *

set_builder(globals())
aimport('ft')

# %%
from typing import Union, Mapping
from exp import *
from util import Ragged
from dictionary import Dictionary

import numpy as np
import pandas as pd
import dask.dataframe as dd


# %%
@task
def csv_dataset():
    return 'data/energodata/*.csv'


# %%
@task
def dataset():
    stop_groups = {'КСУ НСИ', 'HR', 'Первая линия сопровождения'}

    ds = raw_dataset()
    ds = ds[ds.Date > pd.to_datetime('2018-01-01')]
    ds = ds[~ds.Title.isna() & ~ds.Text.isna()]
    ds = ds[ds.Title != 'SAPNSI_ETP']
    ds = ds[~ds.Group.isin(stop_groups)]
    ds['Group'] = ds.Group.map(lambda g: 'Сигма' if g == 'АСУИИиК' else g).astype('category')
    return ds


# %%
@task
def get_groups() -> Mapping[str, Union[int, float]]:
    return {
        'НСИ': 1,
        'МНСИ': 1,
        'АС УиО SAPFI': 2,
        'АС УиО SAPAA': 2,
        'АС УиО SAPCO': 2,
        'АС УиО SAPNU': 2,
        'АСУ-Казначейство': 2,
        'КИСУ Закупки': 2,
        'ЦИУС-ЗУП': 2,
        'Внутренней ИТ-инфраструктуры': 2,
        'Сигма': 4
    }


# %%
@task
def get_labels() -> pd.Series:
    groups = get_groups()
    ds = lite_dataset()
    labels = ds['Group'].copy()
    labels.cat.add_categories('other', inplace=True)
    labels[~labels.isin(groups)] = 'other'
    return labels


# %%
@task
def train_test_split():
    length = len(get_labels())
    rng = np.random.default_rng(0)
    idx = rng.permutation(length)
    train_size = round(length * 0.9)
    test_size = length - train_size
    return idx[0:train_size], idx[test_size:]


# %%
@task
def make_features():
    ds = dataset()
    labels = get_labels()
    train_idx, test_idx = train_test_split()
    text = ds.Title + '\n\n' + ds.Text  # type: dd.Series

    dic = Dictionary()
    train_set = set(train_idx)
    for idx, doc in enumerate(text):
        if idx in train_set:
            dic.fit(doc)
    dic.limit()

    tokens = text.map(lambda t: list(dic(t))).compute()  # type: pd.Series
    lengths = tokens.map(len)

    def ragged(subset):
        sizes = lengths.iloc[subset]
        x = np.empty(sizes.sum(), dtype=np.int32)
        s = 0
        for doc in tokens.iloc[subset]:
            e = s + len(doc)
            x[s:e] = doc
            s = e
        return Ragged(x, sizes.to_numpy())

    return ragged(train_idx), labels.iloc[train_idx], ragged(test_idx), labels.iloc[test_idx], dic


# %%
def save_features(filename: str):
    import joblib
    features = make_features()
    joblib.dump(features, temp(filename))
