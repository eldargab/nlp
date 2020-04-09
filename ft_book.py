#%%
from ipython_setup import aimport
from builder import *

set_builder(globals())
aimport('ft')

# %%
from typing import Union, Mapping, Tuple, Any
from exp import *
from dictionary import Dictionary

import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import util
import ft


@task
def experiment_name():
    return 'fasttext-energodata-2018'


# %%
@task
def raw_dataset():
    def to_parquet(out):
        dd.read_csv(
            'data/energodata/*.csv',
            blocksize=None,
            sample=False,
            parse_dates=['Date'],
            cache_dates=True,
            dtype={'Group': pd.CategoricalDtype()}
        ).to_parquet(out)

    file = output('energodata.parquet', to_parquet)
    return dd.read_parquet(file)


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


@task
def lite_dataset() -> pd.DataFrame:
    ds = dataset()
    return ds[['Date', 'Group']].compute()


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
    ds = dataset()
    labels = ds.Group.compute()  # type: pd.Series
    labels.cat.add_categories('other', inplace=True)
    labels[~labels.isin(get_groups())] = 'other'
    labels.cat.remove_unused_categories(inplace=True)
    return labels


# %%
X = util.RaggedPaddedBatches
Y = pd.Series


@task
def make_features() -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
    ds = dataset()
    text = ds.Title + '\n\n' + ds.Text  # type: dd.Series

    dic = Dictionary()
    test_set = set()
    rng = np.random.default_rng(0)
    for idx, doc in text.iteritems():
        is_test = rng.choice([0, 1], p=[0.9, 0.1])
        if is_test:
            test_set.add(idx)
        else:
            dic.fit(doc)
    dic.limit()

    df = pd.DataFrame.from_dict({
        'X': text.map(lambda t: list(dic(t))).compute(),
        'Y': get_labels()
    })

    df['X_len'] = df.X.map(len)
    df.sort_values(by='X_len', inplace=True)
    del df['X_len']

    test_data = df[df.index.isin(test_set)]
    df.drop(test_set, inplace=True)
    train_data = df

    def to_ragged(data):
        x = util.RaggedPaddedBatches.from_list(data.X.values, batch_size=5)
        return x, data.Y[0:x.size]

    return to_ragged(train_data), to_ragged(test_data), dic


def get_features() -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
    return read_features() or make_features()



# %%
@task
def train_model() -> ft.FastText:
    (x_train, y_train), _, dic = get_features()

    y = torch.tensor(y_train.cat.codes.to_numpy(), dtype=torch.long)

    model = ft.FastText(dict_size=dic.size, dict_dim=100, n_labels=len(y_train.cat.categories), padding_idx=0)

    ft.train(model, x_train.shuffled_tensor_batches(y=y, dtype=torch.long))

    return model


# %%
def predict_prob(x: X):
    model = train_model()
    return torch.cat([ft.predict_prob(model, b) for b in x.tensor_batches(dtype=torch.long)]).numpy()


def get_label_cat_type() -> pd.CategoricalDtype:
    _, y_test = get_features()[1]
    return y_test.dtype


@task
def get_penalties():
    groups = get_groups()
    cat_type = get_label_cat_type()
    penalty_series = cat_type.categories.map(lambda g: groups.get(g, 0))  # type: pd.Series
    return penalty_series.to_numpy(dtype=np.float)


def predict(x: X):
    prob = predict_prob(x)
    penalties = get_penalties()
    score = prob + (prob - 1) * penalties
    y = score.argmax(axis=1)
    return pd.Categorical.from_codes(y, dtype=get_label_cat_type())


def performance(data: Tuple[X, Y]):
    x_test, y_test = data
    guess = predict(x_test)
    return util.describe_results(y_test, guess)


def test_set_performance():
    data = get_features()[1]
    return performance(data)

# %%
test_set_performance()