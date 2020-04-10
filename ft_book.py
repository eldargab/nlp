#%%
from ipython_setup import aimport
from builder import *

set_builder(globals())
aimport('ft')

# %%
from typing import Union, Mapping, Tuple
from dictionary import Dictionary

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask
import util
import ft

# %%
@task
def raw_dataset() -> dd.DataFrame:
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


# %%
@task
def get_groups() -> Mapping[str, float]:
    return {
        'НСИ': 4,
        'МНСИ': 4,
        'АС УиО SAPFI': 9,
        'АС УиО SAPAA': 9,
        'АС УиО SAPCO': 9,
        'АС УиО SAPNU': 9,
        'АСУ-Казначейство': 9,
        'КИСУ Закупки': 9,
        'ЦИУС-ЗУП': 9,
        'Внутренней ИТ-инфраструктуры': 19,
        'Сигма': 99
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
@task
def vectorized_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dictionary]:
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
        'X': text.map(lambda t: np.fromiter(dic(t), dtype=np.int32)).compute(),
        'Y': get_labels()
    })

    test_data = df[df.index.isin(test_set)]

    df.drop(test_set, inplace=True)
    df = df.sample(frac=1)

    return df, test_data, dic


# %%
X = pd.Series
Y = pd.Series

@task
def get_features() -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
    train_data, test_data, dic = vectorized_dataset()
    return (train_data.X, train_data.Y), (test_data.X, test_data.Y), dic


def get_label_cat_type() -> pd.CategoricalDtype:
    _, y_test = get_features()[1]
    return y_test.dtype


def get_labels_count():
    return len(get_label_cat_type().categories)


# %%
@task
def train_model():
    import dask.multiprocessing

    reg_src_module(ft)

    (x_train, y_train), _, dic = get_features()
    n_labels = get_labels_count()

    x = dask.delayed(x_train.to_numpy())
    y = dask.delayed(y_train.cat.codes.to_numpy())
    train = dask.delayed(ft.train)

    with dask.config.set(scheduler='processes'):
        models = dask.compute([train(x, y, dic.size, n_labels, idx, 10) for idx in range(0, 10)])[0]

    return util.EnsembleModel(models)
    # return ft.train(x_train.to_numpy(), y_train.cat.codes.to_numpy(), dic.size, n_labels, 0, 10)


# %%
def predict_prob(x: X) -> np.ndarray:
    model = train_model()
    return model.predict_prob(x)


@task
def get_penalties() -> np.ndarray:
    groups = get_groups()
    cat_type = get_label_cat_type()
    penalty_series = cat_type.categories.map(lambda g: groups.get(g, 0))  # type: pd.Series
    return penalty_series.to_numpy(dtype=np.float)


def predict(x: X):
    prob = predict_prob(x)
    # penalties = get_penalties()
    # score = prob + (prob - 1) * penalties
    score = prob
    y = score.argmax(axis=1)
    return pd.Categorical.from_codes(y, dtype=get_label_cat_type())


def performance(data: Tuple[X, Y]):
    x_test, y_test = data
    guess = predict(x_test)
    return util.describe_results(y_test, guess)


@task
def test_set_performance():
    data = get_features()[1]
    return performance(data)


@task
def train_set_performance():
    data = get_features()[0]
    return performance(data)


# %%
@task
def lite_dataset() -> pd.DataFrame:
    ds = dataset()
    return ds[['Date', 'Group']].compute()

# %%
test_set_performance()
