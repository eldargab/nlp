from builder import *


@task
def experiment_name():
    raise NotImplementedError()


@task
def make_features():
    raise NotImplementedError()


@task
def get_features():
    import joblib
    cache = temp(experiment_name() + '.features.pkl')
    try:
        return joblib.load(cache)
    except FileNotFoundError:
        return make_features()


def save_features():
    import joblib
    features = make_features()
    cache = temp(experiment_name() + '.features.pkl')
    joblib.dump(features, cache)
    reset_builder()
