__version__ = "0.0.22"
print(' Moving to python3.9',__version__)
import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
print(PACKAGEDIR)
from .functions.io import *
#from .eleanor import eleanor
from .My_catalog import mycatalog
#from .functions import eleanor_patch as eleanor 
from .functions.filters import Filters