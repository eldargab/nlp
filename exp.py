from builder import *
import pandas as pd
import dask.dataframe as dd
import util
import glob
import os


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
    csv_glob = csv_dataset()

    csv_files = glob.glob(csv_glob)
    if not csv_files:
        raise RuntimeError('No files matched glob ' + csv_glob)

    # for csv_file in csv_files:
    #     reg_src(csv_file)

    if len(csv_files) > 1:
        filename = os.path.dirname(csv_files[0])
    else:
        filename = csv_files[0]

    filename = os.path.basename(os.path.splitext(filename)[0]) + '.parquet'

    def to_parquet(out):
        dd.read_csv(csv_files, blocksize=None, sample=False, **options).to_parquet(out, compression='gzip')

    return output(filename, to_parquet)


@task
def raw_dataset() -> dd.DataFrame:
    return dd.read_parquet(parquet_dataset())


@task
def dataset() -> dd.DataFrame:
    return raw_dataset()


@task
def lite_dataset() -> pd.DataFrame:
    ds = dataset()
    return ds[['Date', 'Group']].compute()


@task
def groups_summary() -> pd.DataFrame:
    ds = lite_dataset()
    return util.groups_summary(ds)
