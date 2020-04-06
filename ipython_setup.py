from IPython import get_ipython


def autoreload():
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 1')


def aimport(s: str):
    get_ipython().magic('aimport ' + s)


autoreload()
aimport('util, exp, builder')
