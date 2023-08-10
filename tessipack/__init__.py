__version__ = "0.0.21"
print(' Moving to python3.9ß ',__version__)
import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
print(PACKAGEDIR)
from .functions.io import *
#from .eleanor import eleanor
from .My_catalog import mycatalog
import eleanor