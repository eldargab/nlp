#%%
from ipython_setup import aimport
aimport('util, exp, builder')

# %%
from exp import *

set_builder(globals())

@task
def csv_dataset():
    return 'data/energodata/*.csv'

# %%
groups_summary()

# %%
