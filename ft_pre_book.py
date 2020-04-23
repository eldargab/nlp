# %%
from ft_book import *
set_builder(globals())

# %%
import pandas as pd


# %%
@task
def get_vectorized_dataset() -> Tuple[pd.DataFrame, Dictionary]:
    ds = energodata_dataset()
    test_set = get_test_set()
    text = ds.Title + '\n\n' + ds.Text  # type: dd.Series

    dic = Dictionary()
    for idx, doc in text.iteritems():
        if idx not in test_set:
            dic.fit(doc)
    dic.limit(min_docs=30)

    Text, Group, Date = dask.compute(text.map(lambda t: np.fromiter(dic(t), dtype=np.int32)), ds.Group, ds.Date)

    df = pd.DataFrame.from_dict({
        'Text': Text,
        'Group': Group,
        'Date': Date
    })

    df.sort_values(by='Date', inplace=True)

    return df, dic


# %%
class TrainingMixin:
    dict_size: int
    dict_dim: int
    n_labels: int

    def __init__(self, embedding: np.ndarray, **kwargs):
        self.embedding = embedding
        super().__init__(dict_size=embedding.shape[0], dict_dim=embedding.shape[1], **kwargs)

    def new_model(self):
        model = ft.FastText(self.dict_size, self.dict_dim, self.n_labels)
        model.embedding = self.embedding
        return model


# %%
@task
def pre_train_word_embeddings():
    ds, dic = get_vectorized_dataset()
    ds = ds[ds.Date < pd.to_datetime('2018-01-01')]

    dict_dim = 100
    embedding = np.random.default_rng().uniform(-1.0 / dict_dim, 1.0 / dict_dim, (dic.size, dict_dim))

    class PreTraining(TrainingMixin, ft.RegressionTraining):
        def __init__(self, **kwargs):
            super().__init__(embedding=embedding, **kwargs)

    for model_idx, (_, data) in enumerate(ds.resample('6M', on='Date')):
        x = data.Text.to_numpy()
        cat = data.Group.cat.remove_unused_categories().cat
        y = cat.codes.to_numpy()
        n_labels = len(cat.categories)
        training = PreTraining(x=x, y=y, n_labels=n_labels)
        model = training.train_best_vset_loss_epochs(model_idx=model_idx + 1)
        embedding = model.embedding

    return embedding


# %%
X = pd.Series
Y = pd.Series

@task
def get_features() -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
    ds = dataset()
    test_set = get_test_set()
    text = ds.Title + '\n\n' + ds.Text  # type: dd.Series

    _, dic = get_vectorized_dataset()

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
    reg_src_module(ft)

    (x_train, y_train), _, dic = get_features()
    precision = get_target_precision()

    embedding = pre_train_word_embeddings()

    class RegressionTraining(TrainingMixin, ft.RegressionTraining):
        def __init__(self):
            super().__init__(
                embedding=embedding.copy(),
                x=x_train.to_numpy(),
                y=y_train.cat.codes.to_numpy(),
                n_labels=len(y_train.cat.categories)
            )

    with dask.config.set(scheduler='processes'):
        return RegressionTraining().train()

# %%