from __future__ import division, print_function

from lightkurve.utils import LightkurveWarning, validate_method


#import copy
#import logging
#import math
#import re
#import warnings

import numpy as np
#from matplotlib import pyplot as plt

import astropy
from astropy.table import Table
from astropy import units as u
from astropy.units import cds
from astropy.convolution import convolve, Box1DKernel

# LombScargle was moved from astropy.stats to astropy.timeseries in AstroPy v3.2
try:
    from astropy.timeseries import LombScargle
    from astropy.timeseries import implementations #for .main._is_regular
except ImportError:
    from astropy.stats import LombScargle
    from astropy.stats.lombscargle import implementations






class LombScarglePeriodogram():
    """Subclass of :class:`Periodogram <lightkurve.periodogram.Periodogram>`
    representing a power spectrum generated using the Lomb Scargle method.
    """
    def __init__(self, *args, **kwargs):
        self._LS_object = kwargs.pop("ls_obj", None)
        self.nterms = kwargs.pop("nterms", 1)
        self.ls_method = kwargs.pop("ls_method", 'fastchi2')
        super(LombScarglePeriodogram, self).__init__(*args, **kwargs)
        self.label='0'
    def __repr__(self):
        return('LombScarglePeriodogram(ID: {})'.format(self.label))


    @staticmethod
    def from_lightcurve(time='',flux='',flux_err='', minimum_frequency=None, time_format='bkjd',maximum_frequency=None,
                        minimum_period=None, maximum_period=None,
                        frequency=None, period=None,
                        nterms=1, nyquist_factor=1, oversample_factor=None,
                        freq_unit=None, normalization="amplitude", ls_method='fast',false_probabilities=False,method='baluev',
                        **kwargs):
        """Creates a Periodogram from a LightCurve using the Lomb-Scargle method.
        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.
        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequency`
        parameter or a custom regular grid of periods using the `period`
        parameter.
        The sampling of the spectrum can be changed using the
        `oversample_factor` parameter. An oversampled spectrum
        (oversample_factor > 1) is useful for displaying the full details
        of the spectrum, allowing the frequencies and amplitudes to be
        measured directly from the plot itself, with no fitting required.
        This is recommended for most applications, with a value of 5 or
        10. On the other hand, an oversample_factor of 1 means the spectrum
        is critically sampled, where every point in the spectrum is
        independent of the others. This may be used when Lorentzians are to
        be fitted to modes in the power spectrum, in cases where the mode
        lifetimes are shorter than the time-base of the data (which is
        sometimes the case for solar-like oscillations). An
        oversample_factor of 1 is suitable for these stars because the
        modes are usually fully resolved. That is, the power from each mode
        is spread over a range of frequencies due to damping.  Hence, any
        small error from measuring mode frequencies by taking the maximum
        of the peak is negligible compared with the intrinsic linewidth of
        the modes.
        The `normalization` parameter will normalize the spectrum to either
        power spectral density ("psd") or amplitude ("amplitude"). Users
        doing asteroseismology on classical pulsators (e.g. delta Scutis)
        typically prefer `normalization="amplitude"` because "amplitude"
        has higher dynamic range (high and low peaks visible
        simultaneously), and we often want to read off amplitudes from the
        plot. If `normalization="amplitude"`, the default value for
        `oversample_factor` is set to 5 and `freq_unit` is 1/day.
        Alternatively, users doing asteroseismology on solar-like
        oscillators tend to prefer `normalization="psd"` because power
        density has a scaled axis that depends on the length of the
        observing time, and is used when we are interested in noise levels
        (e.g. granulation) and are looking at damped oscillations. If
        `normalization="psd"`, the default value for `oversample_factor` is
        set to 1 and `freq_unit` is set to microHz.  Default values of
        `freq_unit` and `oversample_factor` can be overridden. See Appendix
        A of Kjeldsen & Bedding, 1995 for a full discussion of
        normalization and measurement of oscillation amplitudes
        (http://adsabs.harvard.edu/abs/1995A%26A...293...87K).
        The parameter nterms controls how many Fourier terms are used in the
        model. Setting the Nyquist_factor to be greater than 1 will sample the
        space beyond the Nyquist frequency, which may introduce aliasing.
        The `freq_unit` parameter allows a request for alternative units in frequency
        space. By default frequency is in (1/day) and power in (amplitude).
        Asteroseismologists for example may want frequency in (microHz)
        in which case they would pass `freq_unit=u.microhertz`.
        By default this method uses the LombScargle 'fast' method, which assumes
        a regular grid. If a regular grid of periods (i.e. an irregular grid ofpsd = readPowerspectrumTxt(kicID)
        frequencies) it will use the 'slow' method. If nterms > 1 is passed, it
        will use the 'fastchi2' method for regular grids, and 'chi2' for
        irregular grids.
        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.
        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        minimum_frequency : float
            If specified, use this minimum frequency rather than one over the
            time baseline.
        maximum_frequency : float
            If specified, use this maximum frequency rather than nyquist_factor
            times the nyquist frequency.
        minimum_period : float
            If specified, use 1./minium_period as the maximum frequency rather
            than nyquist_factor times the nyquist frequency.
        maximum_period : float
            If specified, use 1./maximum_period as the minimum frequency rather
            than one over the time baseline.
        frequency :  array-like
            The grid of frequencies to use. If given a unit, it is converted to
            units of freq_unit. If not, it is assumed to be in units of
            freq_unit. This over rides any set frequency limits.
        period : array-like
            The grid of periods to use (as 1/period). If given a unit, it is
            converted to units of freq_unit. If not, it is assumed to be in
            units of 1/freq_unit. This overrides any set period limits.
        nterms : int
            Default 1. Number of terms to use in the Fourier fit.
        nyquist_factor : int
            Default 1. The multiple of the average Nyquist frequency. Is
            overriden by maximum_frequency (or minimum period).
        oversample_factor : int
            Default: None. The frequency spacing, determined by the time
            baseline of the lightcurve, is divided by this factor, oversampling
            the frequency space. This parameter is identical to the
            samples_per_peak parameter in astropy.LombScargle(). If
            normalization='amplitude', oversample_factor will be set to 5. If
            normalization='psd', it will be 1. These defaults can be
            overridden.
         freq_unit : `astropy.units.core.CompositeUnit`
            Default: None. The desired frequency units for the Lomb Scargle
            periodogram. This implies that 1/freq_unit is the units for period.
            With default normalization ('amplitude'), the freq_unit is set to
            1/day, which can be overridden. 'psd' normalization will set
            freq_unit to microhertz.
        normalization : 'psd' or 'amplitude'
            Default: `'amplitude'`. The desired normalization of the spectrum.
            Can be either power spectral density (`'psd'`) or amplitude
            (`'amplitude'`).
        ls_method : str
            Default: `'fast'`. Passed to the `method` keyword of
            `astropy.stats.LombScargle()`.
        kwargs : dict
            Keyword arguments passed to `astropy.stats.LombScargle()`
        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        # Input validation
        normalization = validate_method(normalization, ['psd', 'amplitude'])

        # Setting default frequency units
        if freq_unit is None:
            freq_unit = 1/u.day if normalization == 'amplitude' else u.microhertz

        # Default oversample factor
        if oversample_factor is None:
            oversample_factor = 5. if normalization == 'amplitude' else 1.

        if "min_period" in kwargs:
            warnings.warn("`min_period` keyword is deprecated, "
                          "please use `minimum_period` instead.",
                          LightkurveWarning)
            minimum_period = kwargs.pop("min_period", None)
        if "max_period" in kwargs:
            warnings.warn("`max_period` keyword is deprecated, "
                          "please use `maximum_period` instead.",
                          LightkurveWarning)
            maximum_period = kwargs.pop("max_period", None)
        if "min_frequency" in kwargs:
            warnings.warn("`min_frequency` keyword is deprecated, "
                          "please use `minimum_frequency` instead.",
                          LightkurveWarning)
            minimum_frequency = kwargs.pop("min_frequency", None)
        if "max_frequency" in kwargs:
            warnings.warn("`max_frequency` keyword is deprecated, "
                          "please use `maximum_frequency` instead.",
                          LightkurveWarning)
            maximum_frequency = kwargs.pop("max_frequency", None)
        if "false_probabilities" in kwargs:
            false_probabilities=kwargs.pop("false_probabilities", False)
            del kwargs["false_probabilities"]
        # Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, minimum_period, maximum_period]):
            default_view = 'period'
        else:
            default_view = 'frequency'

        # If period and frequency keywords have both been set, throw an error
        if (not all(b is None for b in [period, minimum_period, maximum_period])) & \
           (not all(b is None for b in [frequency, minimum_frequency, maximum_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. '
                             'Please only use one.')
        flux_quantity=flux
        if (~np.isfinite(flux)).any():
            raise ValueError('Lightcurve contains NaN values. Use lc.remove_nans()'
                             ' to remove NaN values from a LightCurve.')


#        if time_format in ['bkjd', 'btjd', 'd', 'days', 'day', None]:

#            time = time.copy() * u.day
#        else:
#            raise NotImplementedError('time in format {} is not supported.'.format(lc.time_format))

        # Approximate Nyquist Frequency and frequency bin width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(time))))
        fs = (1./(time[-1] - time[0])) / oversample_factor

        # Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)
        print('Nyquist:' ,nyquist,fs)
        # Warn if there is confusing input
        if (frequency is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            log.warning("You have passed both a grid of frequencies "
                        "and min_frequency/maximum_frequency arguments; "
                        "the latter will be ignored.")
        if (period is not None) & (any([a is not None for a in [minimum_period, maximum_period]])):
            log.warning("You have passed a grid of periods "
                        "and minimum_period/maximum_period arguments; "
                        "the latter will be ignored.")

        # Tidy up the period stuff...
        if maximum_period is not None:
            # minimum_frequency MUST be none by this point.
            minimum_frequency = 1. / maximum_period
        if minimum_period is not None:
            # maximum_frequency MUST be none by this point.
            maximum_frequency = 1. / minimum_period
        # If the user specified a period, copy it into the frequency.
        if (period is not None):
            frequency = 1. / period

        # Do unit conversions if user input min/max frequency or period
        if frequency is None:
            if minimum_frequency is not None:
                minimum_frequency = u.Quantity(minimum_frequency, freq_unit)
            if maximum_frequency is not None:
                maximum_frequency = u.Quantity(maximum_frequency, freq_unit)
            if (minimum_frequency is not None) & (maximum_frequency is not None):
                if (minimum_frequency > maximum_frequency):
                    if default_view == 'frequency':
                        raise ValueError('minimum_frequency cannot be larger than maximum_frequency')
                    if default_view == 'period':
                        raise ValueError('minimum_period cannot be larger than maximum_period')
            # If nothing has been passed in, set them to the defaults
            if minimum_frequency is None:
                minimum_frequency = fs
            if maximum_frequency is None:
                maximum_frequency = nyquist * nyquist_factor

            # Create frequency grid evenly spaced in frequency
            frequency = np.arange(minimum_frequency.value, maximum_frequency.value, fs.to(freq_unit).value)

        # Convert to desired units
        frequency = u.Quantity(frequency, freq_unit)

        # Change to compatible ls method if sampling not even in frequency
        if not implementations.main._is_regular(frequency) and ls_method in ['fastchi2','fast']:
            oldmethod = ls_method
            ls_method = {'fastchi2':'chi2','fast':'slow'}[ls_method]
            log.warning("The requested periodogram is not evenly sampled in frequency.\n"
                        "Method has been changed from '{}' to '{}' to allow for this.".format(oldmethod,ls_method))

        if (nterms > 1) and (ls_method not in ['fastchi2', 'chi2']):
            warnings.warn("Building a Lomb Scargle Periodogram using the `slow` method. "
                            "`nterms` has been set to >1, however this is not supported under the `{}` method. "
                            "To run with higher nterms, set `ls_method` to either 'fastchi2', or 'chi2'. "
                            "Please refer to the `astropy.timeseries.periodogram.LombScargle` documentation.".format(ls_method),
                          LightkurveWarning)
            nterms = 1
        # print(kwargs)
        if float(astropy.__version__[0]) >= 3:
            LS = LombScargle(time, flux_quantity,
                             nterms=nterms, normalization='psd', **kwargs)
            power = LS.power(frequency, method=ls_method)
        else:
            LS = LombScargle(time, flux_quantity,
                             nterms=nterms, **kwargs)
            power = LS.power(frequency, method=ls_method, normalization='psd')

        if false_probabilities:
            probability=LS.false_alarm_probability(power,method=method)

        if normalization == 'psd':  # Power spectral density
            # Rescale from the unnormalized power output by Astropy's
            # Lomb-Scargle function to units of flux_variance / [frequency unit]
            # that may be of more interest for asteroseismology.
            power *=  2. / (len(time) * oversample_factor * fs)
        elif normalization == 'amplitude':
            power = np.sqrt(power) * np.sqrt(4./len(time))

        if false_probabilities:
            return frequency,power,probability
        else:

            return frequency,power
        # Periodogram needs properties
        #return LombScarglePeriodogram(frequency=frequency, power=power, nyquist=nyquist,
        #                              targetid=lc.targetid, label=lc.label,
        #                              default_view=default_view, ls_obj=LS,
        #                              nterms=nterms, ls_method=ls_method)

    def model(self, time, frequency=None):
        """Obtain the flux model for a given frequency and time
        Parameters
        ----------
        time : np.ndarray
            Time points to evaluate model.
        frequency : frequency to evaluate model. Default is the frequency at
                    max power.
        Returns
        -------
        result : lightkurve.LightCurve
            Model object with the time and flux model
        """
        if self._LS_object is None:
            raise ValueError('No `astropy` Lomb Scargle object exists.')
        if frequency is None:
            frequency = self.frequency_at_max_power
        f = self._LS_object.model(time, frequency)
        return LightCurve(time, f, label='LS Model', meta={'frequency':frequency},
                            targetid='{} LS Model'.format(self.targetid)).normalize()


def read_file(filename=''):
    from astropy import units as u
    from astropy.table import Table
    import pandas as pd
    data=pd.read_csv(filename)

    astro=Table.from_pandas(data)
    astro['time'].unit=u.d
    astro['flux'].unit=u.electron/u.s
    astro['flux_err'].unit=u.electron/u.s

    return astro


def get_freq_power(filename=None,oversample_factor=1):
    mydata=read_file(filename=filename)
    flux_err=mydata['flux_err'].quantity
    time=mydata['time'].quantity
    flux=mydata['flux'].quantity
    f,p=LombScarglePeriodogram.from_lightcurve(time=time,flux=flux,
                flux_err=flux_err,freq_unit=u.uHz,normalization='psd',oversample_factor=oversample_factor)
    return f,p

def write_freq_power(filename='',freq='',power=''):
    import pandas as pd
    data=pd.DataFrame()
    data['frequency']=freq
    data['power']=power
    data.to_csv(filename,index=False)
    return data
