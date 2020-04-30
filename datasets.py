from builder import App, task
import pandas as pd
import dask.dataframe as dd


class Energodata(App):
    @task
    def energodata_parquet(self) -> str:
        def to_parquet(out):
            ds = dd.read_csv(
                'data/energodata/*.csv',
                blocksize=None,
                sample=False,
                parse_dates=['Date'],
                cache_dates=True,
                dtype={'Group': pd.CategoricalDtype()}
            )
            ds = ds.set_index('Id')
            ds = ds[~ds.Title.isna() & ~ds.Text.isna()]
            ds = ds[ds.Title != 'SAPNSI_ETP']
            ds = ds[ds.Group != 'КСУ НСИ']
            ds.to_parquet(out)

        return self.output('energodata.parquet', to_parquet)

    @task
    def energodata_dataset(self) -> dd.DataFrame:
        return dd.read_parquet(self.energodata_parquet())

    @task
    def energodata_lite(self) -> pd.DataFrame:
        file = self.energodata_parquet()
        return pd.read_parquet(file, columns=['Date', 'Group'])



class Energodata2018(Energodata):
    @task
    def groups(self):
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

    @task
    def dataset(self) -> dd.DataFrame:
        groups = self.groups()
        stop_groups = {'HR', 'Первая линия сопровождения'}

        ds = self.energodata_dataset()
        ds = ds[ds.Date > pd.to_datetime('2018-01-01')]
        ds = ds[~ds.Group.isin(stop_groups)]

        def map_group(g: str):
            if g == 'АСУИИиК':
                g = 'Сигма'
            return g if g in groups else 'other'

        ds['Group'] = ds.Group.map(map_group).astype('category')
        return ds
