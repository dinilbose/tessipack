import astropy
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
from scipy import stats

from astropy.io.votable import parse

def votable_to_pandas(votable_file):
    votable = parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

def extract_essential_wcs_postcard(tpf,header=False):
    '''
    Extract and correct the essential wcs solution and returns corrected wcs solution

    Input: is eleanor postcard
    Output: Corrected wcs solutions
    '''
    from astropy.io.fits import Undefined
    from astropy.wcs import WCS
    wcs_keywords = {'CTYPE1': 'CTYPE1',
                    'CTYPE2': 'CTYPE2',
#                      'CEN_RA': 'CRVAL1',
#                      'CEN_DEC': 'CRVAL2',
                    'CD1_1': 'PC1_1',
                    'CD1_2': 'PC1_2',
                    'CD2_1': 'PC2_1',
                    'CD2_2': 'PC2_2',
                    'TPF_H': 'NAXIS1',
                    'TPF_W': 'NAXIS2'}
    mywcs = {}
    if header==False:
        for oldkey, newkey in wcs_keywords.items():
            if (tpf.header[oldkey] != Undefined):
                mywcs[newkey] = tpf.header[oldkey]
        mywcs['CUNIT1']='deg'
        mywcs['CUNIT2']='deg'
        mywcs['CDELT1']=1.0
        mywcs['CDELT2']=1.0
        mywcs['CTYPE1']='RA---TAN'
        mywcs['CTYPE2']='DEC--TAN'
        mywcs['CRVAL1']=tpf.header[('CEN_RA',1)]
        mywcs['CRVAL2']=tpf.header[('CEN_DEC',1)]
        mywcs['CRPIX1']=float(mywcs['NAXIS1']+1)/2
        mywcs['CRPIX2']=float(mywcs['NAXIS2']+1)/2
    if header==True:
        for oldkey, newkey in wcs_keywords.items():
            if (tpf[oldkey] != Undefined):
                mywcs[newkey] = tpf[oldkey]
        mywcs['CUNIT1']='deg'
        mywcs['CUNIT2']='deg'
        mywcs['CDELT1']=1.0
        mywcs['CDELT2']=1.0
        mywcs['CTYPE1']='RA---TAN'
        mywcs['CTYPE2']='DEC--TAN'
        mywcs['CRVAL1']=tpf[('CEN_RA',1)]
        mywcs['CRVAL2']=tpf[('CEN_DEC',1)]
        mywcs['CRPIX1']=float(mywcs['NAXIS1']+1)/2
        mywcs['CRPIX2']=float(mywcs['NAXIS2']+1)/2
    return WCS(mywcs)

def radec2coord(ra=None,dec=None,unit='degree',frame='icrs'):
    '''Small funtions to convert ra and dec to astropy coord'''
    from astropy import units as u
    if unit=='degree':
        unit=u.degree
        coord = SkyCoord(ra=ra*unit, dec=dec*unit, frame=frame)
    return coord

def radec2pixel(wcs='',ra=None,dec=None,exact=False):
    '''Convert ra dec to pixel'''
    pixel_array=np.empty([0,2])
    if exact==False:
        pixel_array=np.empty([0,2],dtype=int)
    for i in range(len(ra)):

        coord=radec2coord(ra=ra[i],dec=dec[i],unit='degree',frame='icrs')
        pixel=astropy.wcs.utils.skycoord_to_pixel(wcs=wcs,coords=coord)
        if exact==False:
            pixel=np.round(pixel)

#             print(pixel)
        pix=np.array([float(pixel[0]),float(pixel[1])])
        pixel_array=np.append(pixel_array,np.array([pix]),axis=0)
    return pixel_array

def pixe2radec(wcs='',x='',y='',aperture=None):
    coord_array=np.empty([0,2])
    if not aperture.any()==None:
        ar=np.where(aperture)
        y=ar[0]
        x=ar[1]
    for i in range(len(y)):
        coord=astropy.wcs.utils.pixel_to_skycoord(xp=x[i], yp=y[i], wcs=wcs, origin=0, mode='all', cls=None)
        ra=coord.ra.value
        dec=coord.dec.value

    for i in range(len(y)):
        as_ar=astropy.wcs.utils.pixel_to_skycoord(xp=x[i], yp=y[i], wcs=wcs, origin=0, mode='all', cls=None)
        ra=as_ar.ra.value
        dec=as_ar.dec.value

        co=np.array([float(ra),float(dec)])
        coord_array=np.append(coord_array,np.array([co]),axis=0)

    return coord_array


def extract_flux_ap(data='',name='',bkg_type='TPF_LEVEL',flux_name='corr_flux'):
    '''Extract flux from custom aperture'''
    print(data.info_aperture[name][bkg_type].keys())
    return data.info_aperture[name][bkg_type][flux_name]

def flux_filter(data='',flux='',start_time=None,end_time=None,sigma=None,program='eleanor',dropna=True,flux_name='flux',order=True):
    data_frame=pd.DataFrame()
    data_frame[flux_name]=flux

    if order:
        data_frame['time']=data.time.byteswap().newbyteorder()
        data_frame[flux_name]=flux.byteswap().newbyteorder()
    if program=='eleanor':
        quality_flag=data.quality==0
        flag=quality_flag
    else:

        quality_flag=data.quality==0
        flag=quality_flag
        print('hii')

    if not start_time==None:
#         data_frame=data_frame[(data_frame.time>start_time)&(data_frame.time<end_time)]
        time_flag=(data.time >start_time)&(data.time <end_time)
        flag=flag*time_flag

    if not sigma==None:
        extra_flag = (data.corr_flux<(data.corr_flux.mean()+sigma*data.corr_flux.std())) & (data.corr_flux>(data.corr_flux.mean()-0.5*data.corr_flux.std()))
#         flag=flag*~extra_flag
    data_frame=data_frame.loc[flag]
    if dropna==True:
        data_frame=data_frame.dropna()
    return data_frame

def pixel2radec(wcs='',x='',y='',aperture=np.array([])):
    coord_array=np.empty([0,2])
    if aperture.any()==True:
        ar=np.where(aperture)
        y=ar[0]
        x=ar[1]
    for i in range(len(y)):
        coord=astropy.wcs.utils.pixel_to_skycoord(xp=x[i], yp=y[i], wcs=wcs, origin=0, mode='all', cls=None)
        ra=coord.ra.value
        dec=coord.dec.value

    for i in range(len(y)):
        as_ar=astropy.wcs.utils.pixel_to_skycoord(xp=x[i], yp=y[i], wcs=wcs, origin=0, mode='all', cls=None)
        ra=as_ar.ra.value
        dec=as_ar.dec.value

        co=np.array([float(ra),float(dec)])
        coord_array=np.append(coord_array,np.array([co]),axis=0)

    return coord_array

from astropy.timeseries import LombScargle

def plot_periodogram(ax=None,limit=[0.1,25,0.0025],x='',y=''):
    f = np.arange(limit[0],limit[1],limit[2])
    yy=y[~np.isnan(y)]
    xx=x[~np.isnan(y)]
    power=LombScargle(t=xx,y=yy).power(f)
    ax.plot(f,power)
#     ax.plot(x,y)
#     print(power)
    return ax


def flux_filter2(data='',flux='',start_time=None,end_time=None,flux_name='flux',zsigma=3):
    data_frame=pd.DataFrame()
    data_frame[flux_name]=flux
    data_frame['time']=data.time
    data_frame = data_frame[data_frame[flux_name].notna()]
    z =np.abs(stats.zscore(data_frame[flux_name]))<zsigma
    data_frame=data_frame[z]
    flag=data_frame.time>=data_frame.time.min()
    extra_flag=flag
    if not start_time==None:
        time_flag=(data_frame.time >start_time)&(data_frame.time <end_time)
        flag=flag*time_flag
    flag=flag
    data_frame=data_frame.loc[flag]

    return data_frame



def flux_filter_type(data='',flux='', time='',func='median',deviation='mad',sigma=3,start_time=None,end_time=None,flux_name='flux',flux_err='None',time_flag=None,keep_length=False):
    ''' Filter light curve '''


    if type(data)==str:
        data_frame=pd.DataFrame()
    else:
        data_frame=data
    if type(data)==str:
        data_frame[flux_name]=flux

    if not type(time)==str:
        data_frame['time']=time
    else:
        data_frame['time']=data.time
    print('Dataframe', len(data_frame))

    # if not flux_err=='None':
    if not type(flux_err)==str:

        print('Flux error provided')
        data_frame['flux_err']=flux_err

    old_data=data

    if not type(time_flag)==type(None):
    #if not time_flag==None:
    # if len(time_flag)<2:
        print('time_flag', 'applied','Length',len(time_flag),len(data_frame))
        #old one with time_flag as a separate files
        data_frame=data_frame[time_flag]
        #unique timeflag
        #data_frame=data_frame.query('time_flag==0')

    data_frame = data_frame[data_frame[flux_name].notna()]
    print(len(data_frame))
    length=len(data_frame)
    if deviation=='std':
        std=data_frame[flux_name].std()
    if deviation=='mad':
        std=data_frame[flux_name].mad()
    if deviation=='mean_abs':
        std=stats.median_absolute_deviation(data_frame[flux_name])

    if func=='zscore':
        z =np.abs(stats.zscore(data_frame[flux_name]))<sigma
        data_frame=data_frame[z]
    if func=='mean':
        mean=data_frame[flux_name].mean()
        data_frame=data_frame[(data_frame[flux_name]<mean+sigma*std)&(data_frame[flux_name]>mean-sigma*std)]
    if func=='median':
        median=data_frame[flux_name].median()
        data_frame=data_frame[(data_frame[flux_name]<median+sigma*std)&(data_frame[flux_name]>median-sigma*std)]

    flag=data_frame.time>=data_frame.time.min()
    extra_flag=flag
    if not start_time==None:
        time_flag=(data_frame.time >start_time)&(data_frame.time <end_time)
        flag=flag*time_flag
    flag=flag
    data_frame=data_frame.loc[flag]

    filter_percent=(len(data_frame)/length)*100
    data_frame['time_flag']=0

    data_frame['filter_percent']=filter_percent

    if keep_length==True:
        print('Length input and output remains same')
        print('Length of old data',len(old_data))
        # data
        for i in data_frame.columns:
            # print(i)
            old_data[i]=data_frame[i]
        data_frame=old_data
    return data_frame

def find_filename(directory=''):
    import glob
    import os.path
    names = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(directory)]
    return names


class difference():
    def __init__(self,variable=''):
        self.variable=variable
    def on_change(self,new_variable):
        if not self.variable==new_variable:
            self.variable=new_variable
            return True
        else:
            return False

class Receiver(object):
    def __init__(self, client):
        self.client = client
        self.received = False
    def receive_call(self, private_key, sender_id, msg_id, mtype, params, extra):
        self.params = params
        self.received = True
        self.client.reply(msg_id, {"samp.status": "samp.ok", "samp.result": {}})
    def receive_notification(self, private_key, sender_id, mtype, params, extra):
        self.params = params
        self.received = True



def bokeh_errorbar(x, y, xerr=None, yerr=None):

    if xerr:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
# fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if not type(yerr)==type(None):
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
      # fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)

    return y_err_x, y_err_y


def pandas_to_astrotable(data='',time='',flux_columns=['psf_flux','pca_flux','corr_flux','flux_err']):
    from astropy import units as u
    from astropy.table import Table

    astro=Table.from_pandas(data)
    astro['time'].unit=u.d

    for i in flux_columns:
        astro[i].unit=u.electron/u.s

    return astro

import os
def remove_file_if_exist(filename,verbose=False):
    if os.path.exists(filename):
        os.remove(filename)
        if verbose==True:
            print(filename,' Removed')
