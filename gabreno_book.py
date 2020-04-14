# %%
from ft_book import *
set_builder(globals())
aimport('gabreno')

# %%
import pandas as pd

# %%
@task
def get_groups() -> Mapping[str, float]:
    return {
        'НСИ': 0.8,
        'МНСИ': 0.8,
        'АС УиО SAPFI': 0.9,
        'АС УиО SAPAA': 0.9,
        'АС УиО SAPCO': 0.9,
        'АС УиО SAPNU': 0.9,
        'АСУ-Казначейство': 0.9,
        'КИСУ Закупки': 0.9,
        'ЦИУС-ЗУП': 0.9,
        'Внутренней ИТ-инфраструктуры': 0.95,
        'Сигма': 0.99
    }


# %%
@task
def get_features():
    ds = dataset()
    y = get_labels()
    test_set = get_test_set()
    test_set_mask = y.index.isin(test_set)
    train_set_mask = ~test_set_mask

    text = (ds.Title + '\n\n' + ds.Text).compute()  # type: pd.Series

    from dictionary import _tokens
    from sklearn.feature_extraction.text import TfidfVectorizer

    dic = TfidfVectorizer(
        analyzer=lambda s: list(_tokens(s)),
        max_df=0.5,
        min_df=10.0/len(text),
        sublinear_tf=True
    )

    x_train = dic.fit_transform(text[train_set_mask].to_numpy())
    y_train = y[train_set_mask]
    x_test = dic.transform(text[test_set_mask])
    y_test = y[test_set_mask]

    return (x_train, y_train), (x_test, y_test), dic


# %%
@task
def train_model():
    import gabreno

    reg_src_module(gabreno)

    x, y = get_features()[0]

    model = gabreno.Classifier(
        precision=list(get_target_precision()),
        default_group_idx=get_default_group_idx()
    )

    model.fit(x, y.cat.codes)

    return model


# %%
test_set_performance()

# %%