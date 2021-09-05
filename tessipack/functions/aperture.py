import pandas as pd
import os
import numpy as np
from ..My_catalog import mycatalog

def filename(**kwargs):
    '''Return filename'''
    # path='/home/dinilbose/PycharmProjects/light/aperture'
    # path='/home/dinilbose/PycharmProjects/light/cluster/Collinder_69/Aperture/Gaia_aperture'

    # name=str(tic)+'_ap.txt'
    # full_name=p
    full_name=mycatalog.filename(**kwargs)
    return full_name


def create_aperture_file(tic='',id_mycatalog='',name='',sector=''):
    '''Create or append custom aperture files for each source '''
    name_f=filename(id_mycatalog=id_mycatalog,name=name,sector=sector)
    open(name_f, 'a').close()
    #return 1

def add_aperture(tic='',ra='',dec='',id_mycatalog='',name='',sector=''):
    name_f=filename(id_mycatalog=id_mycatalog,name=name,sector=sector)
    # print(name_f)
    exists = os.path.isfile(name_f)
    if not exists:
        create_aperture_file(id_mycatalog=id_mycatalog,name=name,sector=sector)
    #else:
        # Keep presets
    coord=str(ra)+','+str(dec)
    file=open(name_f, 'a')
    file.write(coord+"\n")
    file.close()

def load_aperture(**kwargs):
    name=filename(**kwargs)
    data=pd.read_csv(name,names=['ra','dec'])
    return data

def create_mask(x=1,y=1,value=False):
    '''Create a mask file'''
    mask=np.zeros([x,y])

    if value==True:
        mask=mask==0
    elif value==False:
        mask=mask!=0
    else:
        mask[mask==0]=value

    return mask

def replace_mask(mask='',x=1,y=1,value=True):
    '''Replace pixels values in a mask file'''
    mask[int(y),int(x)]=value
    return mask
