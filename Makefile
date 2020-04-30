ipython:
	@ipython -i -c "\
	import numpy as np;\
	import pandas as pd;\
	import dask;\
	import dask.dataframe as dd;\
	import itertools;\
	import ipython_setup;\
	"


.PHONY: ipython
