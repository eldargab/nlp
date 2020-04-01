ipython:
	@ipython -i --ext autoreload -c "\
	get_ipython().magic('autoreload 1');\
	import numpy as np;\
	import torch;\
	import itertools;\
	import dask;\
	import dask.dataframe as dd;\
	from builder import *;\
	set_default_builder(globals());\
	get_ipython().magic('aimport util, exp');\
	"


.PHONY: ipython
