__version__ = "0.0.1"
print('Under development.....')
import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
print(PACKAGEDIR)
from .functions.io import *
from .eleanor import eleanor
from .My_catalog import mycatalog
