from builder import task
import util
import pandas as pd


@task
def train_model():
    raise NotImplementedError()


@task
def get_features():
    raise NotImplementedError()


@task
def get_model():
    return train_model()


def get_label_cat_type() -> pd.CategoricalDtype:
    _, y_test = get_features()[1]
    return y_test.dtype


def get_labels_count():
    return len(get_label_cat_type().categories)


def predict(x):
    model = get_model()
    y = model.predict(x)
    return pd.Categorical.from_codes(y, dtype=get_label_cat_type())


def performance(data):
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
