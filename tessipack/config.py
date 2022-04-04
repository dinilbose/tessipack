import os
home=os.path.expanduser('~')
config_file='config_tessipack.py'
config_path=home+'/'+config_file
from pathlib import Path
path = Path(config_path)
import sys

if not path.is_file():

    #Location of the Catalog
    home_folder='/home/dinilbose/data_tessipack/'
    #Name of the catalog file
    catalog_name='Mycatalog_v1.csv'
    catalog_path=home_folder+'/'+catalog_name
    #Location of light curves & apertures for individual stars
    data_folder='/home/dinilbose/data_tessipack/data/'
    #Location to save tess files required for light curve processing
    extra_data=home_folder+'other_data/extra_data/'

else:
    sys.path.append(home)
    from config_tessipack import *

#check for home folder else copy data and files
path=Path(home_folder)

if not os.path.is_dir()
    print('Intial files not found')
    #shutil.copytree(data_folder, path, dirs_exist_ok=False)

    print('Copying intial files')


    print('Files copied to ',)
