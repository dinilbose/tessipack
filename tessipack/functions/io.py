# This file is for input output operation Like reading a file extra
from ..My_catalog import mycatalog
import numpy as np
import pandas as pd
from ..functions import flags
from ..functions import utils
import os


def read_lightcurve(name='eleanor_flux',id_mycatalog='',sector='',extra_flag_file=None,sigma=5,flux_name='pca_flux',filter=True,time_flag='default',keep_length=True,func='median',deviation='mad',):
    '''Function for reading a light curve from set of files
    name: 'eleanor_flux' or 'eleanor_flux_current'
    extra_flag_file:Supply extra flag file name or path
    sigma=cuttoff
    func,deviation:Check utils flux_filter_type for examples
    Keep_length: Data frame length remains same after filter
    '''
    clus=mycatalog.pointer(catalog='mycatalog',id_mycatalog=id_mycatalog).cluster.values[0]
    if name=='eleanor_flux':

        Data=pd.read_csv(mycatalog.filename(name='eleanor_flux',id_mycatalog=id_mycatalog,sector=sector))

    if name=='eleanor_flux_current':

        Data=pd.read_csv(mycatalog.filename(name='eleanor_flux_current',id_mycatalog=id_mycatalog,sector=sector))

    if not type(extra_flag_file)==type(None):

        print('Check for errors')
        Data=flags.apply_flag(filename=extra_flag_file,data=Data,cluster=clus,apply_cluster=True)

    elif type('extra_flag_file')==type(np.array([])):

        print('Please correct the code')

    if filter==True:

        if time_flag=='default':

            time_flag_frame=pd.read_csv(mycatalog.filename(name='eleanor_time_flag',id_mycatalog=id_mycatalog,sector=sector))

            Data=utils.flux_filter_type(time_flag=time_flag_frame.time_flag.values,data=Data,func='median',deviation='mad',flux_name=flux_name,sigma=sigma,keep_length=keep_length).reset_index(drop=True)

        else:

            Data=utils.flux_filter_type(data=Data,func=func,deviation=deviation,flux_name=flux_name,sigma=sigma,keep_length=keep_length).reset_index(drop=True)

    Data = Data.loc[:, ~Data.columns.str.contains('^Unnamed')]

    return Data

def read_prtable(id_mycatalog='',sector=''):

    f=mycatalog.filename(id_mycatalog=id_mycatalog,name='bokeh_periodogram_table',sector=sector)

    if os.path.exists(f):
        f=mycatalog.filename(id_mycatalog=id_mycatalog,name='bokeh_periodogram_table',sector=sector)
        data=pd.read_csv(f)
    else:
        data=pd.DataFrame(columns=['x','y'])

    return data
