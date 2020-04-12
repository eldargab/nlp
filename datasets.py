from builder import task, output

import dask.dataframe as dd
import pandas as pd


def _energodata_parquet():
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

    return output('energodata.parquet', to_parquet)


@task
def energodata_dataset() -> dd.DataFrame:
    file = _energodata_parquet()
    return dd.read_parquet(file)


@task
def energodata_lite() -> pd.DataFrame:
    file = _energodata_parquet()
    return pd.read_parquet(file, columns=['Date', 'Group'])
