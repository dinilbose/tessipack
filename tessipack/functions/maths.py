import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from ..functions import lomb
from astropy import units as u
from astropy.table import Table
from ..My_catalog import mycatalog
from ..functions import utils
from astropy import units
from scipy.optimize import minimize_scalar
import numpy as np
import scipy
from scipy import interpolate

def lomb_scargle(id_gaia='',id_mycatalog='',flux_name='pca_flux',sigma=5,time='',flux='',flux_err='',oversample_factor=2,false_probabilities=False,normalization='psd',**kwargs):
    f=[]
    p=[]
    if not id_gaia=='' or not id_mycatalog=='' :

        my_file=mycatalog.filename(name='eleanor_flux',id_gaia=id_gaia,id_mycatalog=id_mycatalog)
        data=pd.read_csv(my_file)
        # flux_name='pca_flux'
        data=utils.flux_filter_type(func='median', deviation='mad', sigma=sigma, data=data, flux=data[flux_name], flux_name=flux_name)
        data=data.dropna(subset = [flux_name])
        astro=Table.from_pandas(data)
        astro['time'].unit=u.d
        astro['pca_flux'].unit=u.electron/u.s
        astro['flux_err'].unit=u.electron/u.s
        time=astro['time'].quantity
        flux=astro['pca_flux'].quantity
        flux_err=astro['flux_err'].quantity
        oversample_factor=oversample_factor
        f,p=lomb.LombScarglePeriodogram.from_lightcurve(time=time,flux=flux,
                    flux_err=flux_err,freq_unit=u.uHz,normalization=normalization,oversample_factor=oversample_factor,**kwargs)

    if not type(flux)==str:

        data=pd.DataFrame()
        data['time']=time
        data['pca_flux']=flux
        data['flux_err']=flux_err
        data=data.dropna(subset = [flux_name])

        astro=Table.from_pandas(data)
        astro['time'].unit=u.d
        astro['pca_flux'].unit=u.electron/u.s
        astro['flux_err'].unit=u.electron/u.s
        time=astro['time'].quantity
        flux=astro['pca_flux'].quantity
        flux_err=astro['flux_err'].quantity
        oversample_factor=oversample_factor
        if false_probabilities:
            print('false probability')
            f,p,n=lomb.LombScarglePeriodogram.from_lightcurve(time=time,flux=flux,
                        flux_err=flux_err,freq_unit=u.uHz,normalization=normalization,oversample_factor=oversample_factor,false_probabilities=True,**kwargs)
            return f,p,n
        else:
            f,p=lomb.LombScarglePeriodogram.from_lightcurve(time=time,flux=flux,
                        flux_err=flux_err,freq_unit=u.uHz,normalization=normalization,oversample_factor=oversample_factor)

    return f, p


def folding(data='',flux_name='',flux='',time='',period='',duplicate=True,addition=0.5,append=False):
    '''Fold the data using the period
    Same function exist in utils also

    '''
    # print('Period',period)
    from astropy import units
    if not type(data)==type(''):
        # m=period*units.microhertz
        T=1/period.to(1/units.d)
        # new = np.mod(data.time, T) / T;
        phase = np.mod(data.time, T.value) / T.value;
        data['phase']=phase

    else:
        print('Please correct the code')
        data=None

    if duplicate==True:

        # data['phase2']=data['phase']
        data['phase2']=np.nan

        # new=data[data.phase<=addition].reset_index()
        # new['phase2']=new['phase']+1
        condition=data.phase<=addition
        data['phase2'][condition]=data['phase'][condition]+1

        condition=data.phase>=addition
        data['phase2'][condition]=data['phase'][condition]-1

    if append==True:

        data2=data.dropna(subset=['phase2'])
        data2['phase']=data2['phase2']
        data=data.append(data2)



    return data


def __fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess,maxfev=10000)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def __set_fit(i=0,data='',flux_name='pca_flux'):
    new=folding(data=data,period=i*units.microhertz).dropna()
    tt = new.phase
    yy = new.pca_flux
    yynoise = yy
    res = __fit_sin(tt, yynoise)
    w=np.sqrt(np.sum((new[flux_name]-res["fitfunc"](tt))**2)/len(tt))
    return w

def minimise_period(data='',x1='',x2=''):
    res = minimize_scalar(__set_fit, args=(data),method='bounded',bounds=(x1,x2))
    return res.x


def interpolate_isochrone(isochrone,variable='M_ini',xnew=None,xdelta=0.001):

    '''Interpolate isochrone...
    provide isochrone as a pandas tabe.. all the header will interpolated using the variable provided
    '''
    x=isochrone[variable]
    new_isochrone=pd.DataFrame()

    if type(xnew)==type(None):
        # print('issue')
        xnew = np.arange(x.min(), x.max(), xdelta)

    header=isochrone.columns
    for col in header:
        # print(col)
        f=''
        y=isochrone[col]
        f = interpolate.interp1d(x, y,fill_value="extrapolate")
        # print(xnew)
        new_isochrone[col]=f(xnew)
    return new_isochrone
