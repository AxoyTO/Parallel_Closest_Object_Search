from functools import partial
import sys

models_dir = "/ModelSet"
print_flushed = partial(print, flush=True)
results_dict = {}

LOAD_OUTPUT = 0
RESULT_OUTPUT = 0
METHOD = 'SCIPY_DH'

if METHOD == 'KDTREE':
    sys.setrecursionlimit(10000)

fixed_model_name = "airplane_0627"