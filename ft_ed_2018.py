from typing import Tuple

from builder import task
from datasets import Energodata2018
from dictionary import Dictionary
from exp import Exp
import pandas as pd
import numpy as np


X = np.ndarray
Y = pd.Series


class FastTextEd2018(Exp, Energodata2018):
    @task
    def features(self) -> Tuple[Tuple[X, Y], Tuple[X, Y], Dictionary]:
        ds = self.dataset().compute()  # type: pd.DataFrame
        text = ds.Title + '\n\n' + ds.Text

        test_set = ds.sample(frac=0.15, random_state=0)
        train_set_mask = ~ds.index.isin(set(test_set.index))

        dic = Dictionary()
        for doc in text[train_set_mask]:
            dic.fit(doc)
        dic.limit()

        x = text.map(lambda t: np.fromiter(dic(t), dtype=np.int32)).to_numpy()

        x_train = x[train_set_mask]
        y_train = ds.Group[train_set_mask]

        x_test = x[~train_set_mask]
        y_test = ds.Group[~train_set_mask]

        return (x_train, y_train), (x_test, y_test), dic


    def model(self):
        import ft

        self.reg_src_module(ft)

        (x, y), _, dic = self.features()
        y = y.cat.codes.to_numpy()

        model = ft.FastText(
            n_labels=self.n_labels(),
            dict_size=dic.size,
            dict_dim=100
        )

        for epoch in range(1, 8):
            loss = 0.0
            for i in range(len(y)):
                loss += model.backward(x[i], y[i], lr=0.2)
            loss = loss / len(y)
            print(f'epoch {epoch}, loss: {loss}')

        return model


    def train_set(self):
        return self.features()[0]


    def test_set(self):
        return self.features()[1]

