import os
home=os.path.expanduser('~')
config_file='config_tessipack.py'
config_path=home+'/'+config_file
from pathlib import Path
config_path = Path(config_path)
import sys
import shutil
from distutils.dir_util import copy_tree

if not config_path.is_file():

    #Location of the home folder
    home_folder=home
    main_folder=home_folder+'/data_tessipack/'
    #Name of the catalog file
    catalog_name='Mycatalog_v1.csv'
    catalog_path=main_folder+catalog_name
    #Location of light curves & apertures for individual stars
    data_folder=main_folder+'data/'
    #Location to save tess files required for light curve processing
    extra_data=main_folder+'other_data/extra_data/new_data/'

else:
    print('Config from user')
    sys.path.append(home)
    from config_tessipack import *

#check for home folder else copy data and files
copy_dir=os.getcwd()+'/data/'
copy_dir=os.path.join(os.path.dirname(__file__))+'/data/'
print(main_folder)
if not os.path.isdir(main_folder):
    #print('Intial files not found')
    print('Files copied from ', copy_dir,' to ', home_folder)

    copy_tree(copy_dir, home_folder)

    print('Copying intial files',)


    print('Files copied from ', copy_dir,' to ', home_folder)
