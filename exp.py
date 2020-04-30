from typing import Any, Tuple, Callable
from builder import task
import pandas as pd
import util


X = Any
Y = pd.Series


class Exp:
    test_set: Callable[[], Tuple[X, Y]]
    train_set: Callable[[], Tuple[X, Y]]
    model: Callable[[], Any]


    @task
    def label_cat_type(self) -> pd.CategoricalDtype:
        _, y_test = self.test_set()
        return y_test.dtype


    @task
    def n_labels(self):
        return len(self.label_cat_type().categories)


    @task
    def default_group_idx(self):
        return self.label_cat_type().categories.get_loc('other')


    def predict(self, x: X):
        model = self.model()
        y = model.predict(x)
        return pd.Categorical.from_codes(y, dtype=self.label_cat_type())


    def performance(self, data: Tuple[X, Y]):
        x_test, y_test = data
        guess = self.predict(x_test)
        return util.describe_results(y_test, guess)


    @task
    def test_set_performance(self):
        data = self.test_set()
        return self.performance(data)


    @task
    def train_set_performance(self):
        data = self.train_set()
        return self.performance(data)
