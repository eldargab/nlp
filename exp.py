from builder import *
import pandas as pd
import dask.dataframe as dd
import util


@task
def csv_dataset() -> str:
    raise NotImplementedError()


@task
def dataset() -> dd.DataFrame:
    import glob
    import os

    csv_glob = csv_dataset()
    src_files = glob.glob(csv_glob)

    if len(src_files) == 0:
        raise RuntimeError(f'No files matching {csv_glob}')

    if len(src_files) == 1:
        cache_basename = src_files[0]
    else:
        cache_basename = os.path.dirname(src_files[0]) + '--' + '-'.join(
            os.path.splitext(os.path.basename(f))[0] for f in src_files
        )

    cache = derive_filename(cache_basename, '.parquet')

    if not is_fresh(cache, *src_files):
        util.read_csv_dataset(csv_glob).to_parquet(cache)

    return dd.read_parquet(cache)


@task
def lite_dataset() -> pd.DataFrame:
    ds = dataset()
    return ds[['Id', 'Date', 'Group']].compute()
