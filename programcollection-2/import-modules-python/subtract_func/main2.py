import os
import sys
from os.path import dirname, join, abspath

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

try:
    from addition_func.main1 import add
    add()
except ModuleNotFoundError:
    print("Module 'addition_func' not found. Please ensure it is installed and located in the parent directory.")
except AttributeError:
    print("Function 'add' not found in the module 'addition_func'.")