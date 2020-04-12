#%%
from ipython_setup import aimport
from builder import *

set_builder(globals())
aimport('ft')

# %%
from typing import Mapping, Tuple
from dictionary import Dictionary
from datasets import *
from exp import *

import numpy as np
import pandas as pd
import dask.dataframe as dd
import util
import ft


# %%
@task
def dataset():
    stop_groups = {'HR', 'Первая линия сопровождения'}

    ds = energodata_dataset()
    ds = ds[ds.Date > pd.to_datetime('2018-01-01')]
    ds = ds[~ds.Group.isin(stop_groups)]
    ds['Group'] = ds.Group.map(lambda g: 'Сигма' if g == 'АСУИИиК' else g).astype('category')
    return ds


# %%
@task
def get_groups() -> Mapping[str, float]:
    return {
        'НСИ': 0,
        'МНСИ': 0,
        'АС УиО SAPFI': 20,
        'АС УиО SAPAA': 5,
        # 'АС УиО SAPCO': 9,
        # 'АС УиО SAPNU': 9,
        'АСУ-Казначейство': 20,
        'КИСУ Закупки': 2,
        'ЦИУС-ЗУП': 2,
        # 'Внутренней ИТ-инфраструктуры': 10,
        # 'Сигма': 200
    }


# %%
@task
def get_labels() -> pd.Series:
    ds = dataset()
    labels = ds.Group.compute()  # type: pd.Series
    labels.cat.add_categories('other', inplace=True)
    labels[~labels.isin(get_groups())] = 'other'
    labels.cat.remove_unused_categories(inplace=True)
    return labels


@task
def get_penalties() -> np.ndarray:
    groups = get_groups()
    cat_type = get_label_cat_type()
    penalty_series = cat_type.categories.map(lambda g: groups.get(g, 0))  # type: pd.Series
    return penalty_series.to_numpy(dtype=np.float)


@task
def get_test_set():
    labels = get_labels()
    labels = labels.sample(frac=0.15, random_state=1)
    return set(labels.index)


# %%
X = pd.Series
Y = pd.Series

@task
def get_features() -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
    ds = dataset()
    test_set = get_test_set()
    text = ds.Title + '\n\n' + ds.Text  # type: dd.Series

    dic = Dictionary()
    for idx, doc in text.iteritems():
        if idx not in test_set:
            dic.fit(doc)
    dic.limit()

    x = text.map(lambda t: np.fromiter(dic(t), dtype=np.int32)).compute()
    y = get_labels()

    test_set_mask = y.index.isin(test_set)
    x_test = x[test_set_mask]
    y_test = y[test_set_mask]

    x_train = x[~test_set_mask]
    y_train = y[~test_set_mask]

    return (x_train, y_train), (x_test, y_test), dic


# %%
@task
def train_model():
    import dask.multiprocessing

    reg_src_module(ft)

    (x_train, y_train), _, dic = get_features()
    n_labels = get_labels_count()
    penalties = get_penalties()

    x = dask.delayed(x_train.to_numpy())
    y = dask.delayed(y_train.cat.codes.to_numpy())
    train = dask.delayed(ft.train)

    with dask.config.set(scheduler='processes'):
        models = dask.compute([train(x, y, dic.size, n_labels, idx) for idx in range(0, 10)])[0]

    return util.EnsembleModel([util.ProbThresholdModel(m, penalties) for m in models])
    # return util.EnsembleModel(models)
    # model = ft.train(x_train.to_numpy(), y_train.cat.codes.to_numpy(), dic.size, n_labels, 0)
    # return util.ProbThresholdModel(model, penalties)
