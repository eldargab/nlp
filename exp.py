from builder import *


@task
def experiment_name():
    raise NotImplementedError()


@task
def make_features():
    raise NotImplementedError()


def read_features():
    import joblib
    cache = temp(experiment_name() + '.features.pkl')
    reg_src(cache)
    try:
        return joblib.load(cache)
    except FileNotFoundError:
        return None


def save_features():
    import joblib
    features = make_features()
    cache = temp(experiment_name() + '.features.pkl')
    joblib.dump(features, cache)
    reset_builder()
