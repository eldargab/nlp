#%%
from builder import *
from exp import *
import dask.dataframe as dd
import pandas as pd
import util


set_default_builder(globals())


@task
def csv_dataset():
    return 'data/energodata/*.csv'


# %%
groups_summary()

# %%