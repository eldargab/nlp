ipython:
	@ipython -i --ext autoreload -c "\
	get_ipython().magic('autoreload 1');\
	import numpy as np;\
	import torch;\
	import itertools;\
	get_ipython().magic('aimport util');\
	"


.PHONY: ipython
