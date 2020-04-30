from IPython import get_ipython


def autoreload():
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 1')


def aimport(s: str):
    get_ipython().magic('aimport ' + s)


def aimport_project_files():
    import glob
    import os

    for f in glob.glob('*.py'):
        f = os.path.basename(f)
        f = os.path.splitext(f)[0]
        if f != 'temp':
            aimport(f)


autoreload()
aimport_project_files()

