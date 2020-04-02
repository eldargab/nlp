from builder import *
import pandas as pd
import dask.dataframe as dd
import util


@task
def csv_dataset() -> str:
    raise NotImplementedError()


@task
def csv_format():
    return dict(
        parse_dates=['Date'],
        cache_dates=True,
        dtype={'Group': pd.CategoricalDtype()}
    )


@task
def parquet_dataset() -> str:
    options = csv_format()

    def to_parquet(files, out):
        dd.read_csv(files, blocksize=None, sample=False, **options).to_parquet(out, compression='gzip')

    return compile(csv_dataset(), '.parquet', to_parquet)


@task
def dataset() -> dd.DataFrame:
    return dd.read_parquet(parquet_dataset())


@task
def lite_dataset() -> pd.DataFrame:
    ds = dataset()
    return ds[['Date', 'Group']].compute()


@task
def groups_summary() -> pd.DataFrame:
    ds = lite_dataset()
    return util.groups_summary(ds)
