"""Defines the Seismology class."""
import logging
import warnings
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from bokeh.models import ColumnDataSource

from astropy import units as u
from astropy.units import cds

from lightkurve import MPLSTYLE
from lightkurve.seismology import utils, stellar_estimators
from lightkurve.periodogram import SNRPeriodogram
from lightkurve.utils  import LightkurveWarning, validate_method
from lightkurve.seismology  import  SeismologyQuantity

from astropy import units
from tessipack.functions import maths
from tessipack.functions import io
import lightkurve
import Periodo
from env import Environment
from astropy import units
from bokeh.models import CustomJS, TextInput, Paragraph
from bokeh.models import Button, Select,CheckboxGroup  # for saving data

# Import the optional Bokeh dependency required by ``interact_echelle```,
# or print a friendly error otherwise.
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure
    from bokeh.models import LogColorMapper, Slider, RangeSlider, Button
    from bokeh.layouts import layout, Spacer
except:
    # Nice error will be raised when ``interact_echelle``` is called.
    pass

log = logging.getLogger(__name__)



class Interactive(Environment):
    env=Environment

    def __init__(self):

        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]
        self.palette="Spectral11"
        self.env.text_osc_query= TextInput(value='n_pg>=0', title="Select Cluster")

        #extra_flag_file='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/extra_flag.flag'
        #extra_flag_file=self.env.extra_flag_file
        #lc=io.read_lightcurve(name='eleanor_flux',id_mycatalog=self.id_mycatalog,sector=self.env.sector,extra_flag_file=extra_flag_file,sigma=5)
        #f,p=maths.lomb_scargle(flux=lc['pca_flux'],time=lc.time,flux_err=lc.flux_err)
        ff,pp=self.read_fits_get_fp()
        f=(ff*u.Hz).to(u.uHz)
        p=pp*u.electron/u.s

        from lightkurve import periodogram as lk_prd_module
        period=lk_prd_module.Periodogram(f,p)
        #period=Periodo.Periodogram(frequency=f,power=p)

        self.periodogram = period
        self.env.minimum_frequency=1
        self.env.maximum_frequency=8000
        self.env.maxdnu=50
        self.env.dnu_val=20
        # self.tb_source.on_change('data',self.update_id)
        #self.env.tb_periodogram.on_change('data',self.update_plot)

        self.env.frequency_minimum_text=TextInput(value=str(self.env.minimum_frequency), title="Frequency_min",width=100)
        self.env.frequency_maximum_text=TextInput(value=str(self.env.maximum_frequency), title="Frequency_max",width=100)
        self.env.frequency_maxdnu_text=TextInput(value=str(self.env.maxdnu), title="Maxdnu",width=100)
        self.env.dnu_text=TextInput(value=str(self.env.dnu_val), title="Delta Nu",width=100)

        self.env.update_int_button = Button(label="Update Plot", button_type="success",width=150)
        self.env.update_int_button.on_click(self.update_value)
        self.interact_echelle()
        #self.mesa_interactive()


    def initialize_dnu_periodogram():
        '''Function initialise the the dnu periodogram table source and also the periodogram graph
        '''



        return tb_other_peridogram, fig_echelle_periodogram


    def read_fits_get_fp(self):

        "Test function: Read fits file and get f and p"
        print('Running read fits')
        from astropy.io import fits
        import pandas
        with fits.open('/Users/dp275303/work/tessipack_developement_test/PSD_003429205_no_gap.fits') as data:
            df = pandas.DataFrame(data[0].data)
        return df[0].values,df[1].values

        

    def update_value(self):
        self.env.minimum_frequency=float(self.env.frequency_minimum_text.value)
        self.env.maximum_frequency=float(self.env.frequency_maximum_text.value)
        self.env.maxdnu=float(self.env.frequency_maxdnu_text.value)
        self.env.dnu_val=float(self.env.dnu_text.value)
        self.env.dnu_slider.value=self.env.dnu_val

        print('Values transfering',self.env.minimum_frequency,self.env.maximum_frequency,self.env.maxdnu)
        self.update_plot(0,0,0)


    def update_plot(self,attr,old,new):

        self.id_mycatalog=self.env.tb_source.data['id_mycatalog'][0]



        #dnu = self.deltanu for current purpose
        dnu=self.dnu_val

        # f=self.env.tb_periodogram.data['x']*units.microhertz
        # p=self.env.tb_periodogram.data['y']*units.microhertz
        #extra_flag_file='/home/dinilbose/PycharmProjects/light_cluster/cluster/Collinder_69/Data/extra_flag.flag'
        # extra_flag_file=self.env.extra_flag_file
        # lc=io.read_lightcurve(name='eleanor_flux',id_mycatalog=self.id_mycatalog,sector=self.env.sector,extra_flag_file=extra_flag_file,sigma=5)
        # f,p=maths.lomb_scargle(flux=lc['pca_flux'],time=lc.time,flux_err=lc.flux_err)


        # ff,pp=self.read_fits_get_fp()
        # f=(ff*u.Hz).to(u.uHz)
        # p=pp*u.electron/u.s
        # from lightkurve import periodogram as lk_prd_module
        # period=lk_prd_module.Periodogram(f,p)

        # period=Periodo.Periodogram(frequency=f,power=p)

        # self.periodogram = period


        ep, x_f, y_f = self._clean_echelle(deltanu=dnu,
                                       minimum_frequency=self.env.minimum_frequency,
                                       maximum_frequency=self.env.maximum_frequency)
        self.env.fig_tpfint.select('img')[0].data_source.data['image'] = [ep.value]
        self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(dnu)



        #print('Test##',self.env.fig_tpfint.select('img').glyph.y)


        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        self.env.fig_tpfint.y_range.start = y_f[0].value
        self.env.fig_tpfint.y_range.end = y_f[-1].value


        self.env.fig_tpfint.select('img').glyph.y=y_f[0].value
        self.env.fig_tpfint.select('img').glyph.dh=y_f[-1].value
        self.env.fig_tpfint.select('img').glyph.color_mapper.low=color_mapper.low
        self.env.fig_tpfint.select('img').glyph.color_mapper.high=color_mapper.high
        self.env.stretch_sliderint.start=vlo
        self.env.stretch_sliderint.end=vhi
        self.env.stretch_sliderint.value=(lo,hi)

        self.env.dnu_slider.start=0.01
        self.env.dnu_slider.end=self.env.maxdnu


    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_deltanu()` first.")
        return deltanu


    def _clean_echelle(self, deltanu=None, numax=None,
                         minimum_frequency=None, maximum_frequency=None,
                         smooth_filter_width=.1, scale='linear'):
        """Takes input seismology object and creates the necessary arrays for an echelle
        diagram. Validates all the inputs.

        Parameters
        ----------
        deltanu : float
            Value for the large frequency separation of the seismic mode
            frequencies in the periodogram. Assumed to have the same units as
            the frequencies, unless given an Astropy unit.
            Is assumed to be in the same units as frequency if not given a unit.
        numax : float
            Value for the frequency of maximum oscillation. If a numax is
            passed, a suitable range one FWHM of the mode envelope either side
            of the will be shown. This is overwritten by custom frequency ranges.
            Is assumed to be in the same units as frequency if not given a unit.
        minimum_frequency : float
            The minimum frequency at which to display the echelle
            Is assumed to be in the same units as frequency if not given a unit.
        maximum_frequency : float
            The maximum frequency at which to display the echelle.
            Is assumed to be in the same units as frequency if not given a unit.
        smooth_filter_width : float
            If given a value, will smooth periodogram used to plot the echelle
            diagram using the periodogram.smooth(method='boxkernel') method with
            a filter width of `smooth_filter_width`. This helps visualise the
            echelle diagram. Is assumed to be in the same units as the
            periodogram frequency.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        scale: str
            Set z axis to be "linear" or "log". Default is linear.

        Returns
        -------
        ep : np.ndarray
            Echelle diagram power
        x_f : np.ndarray
            frequencies for X axis
        y_f : np.ndarray
            frequencies for Y axis
        """
        if (minimum_frequency is None) & (maximum_frequency is None):
            numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)

        if (not hasattr(numax, 'unit')) & (numax is not None):
            numax = numax * self.periodogram.frequency.unit
        if (not hasattr(deltanu, 'unit')) & (deltanu is not None):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
            power = pgsmooth.power     # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable
            power = self.periodogram.power     # Makes code below more readable

        fmin = freq[0]
        fmax = freq[-1]

        # Check for any superfluous input
        if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            warnings.warn("You have passed both a numax and a frequency limit. "
                          "The frequency limit will override the numax input.",
                          LightkurveWarning)

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, freq.unit).value
            if numax > freq[-1].value:
                raise ValueError("You can't pass in a numax outside the"
                                "frequency range of the periodogram.")

            fwhm = utils.get_fwhm(self.periodogram, numax)

            fmin = numax - 2*fwhm
            if fmin < freq[0].value:
                fmin = freq[0].value

            fmax = numax + 2*fwhm
            if fmax > freq[-1].value:
                fmax = freq[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin =  u.Quantity(minimum_frequency, freq.unit).value
            if fmin > freq[-1].value:
                raise ValueError('Fmin',fmin,"You can't pass in a limit outside the "
                                 "frequency range of the periodogram.")

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, freq.unit).value
            if fmax > freq[-1].value:
                raise ValueError('Fmax',fmax,"You can't pass in a limit outside the "
                                 "frequency range of the periodogram.")

        # Make sure fmin and fmax are Quantities or code below will break
        fmin = u.Quantity(fmin, freq.unit)
        fmax = u.Quantity(fmax, freq.unit)

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < freq[-1] - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(freq))
        x0 = int(freq[0] / fs)

        ff = freq[int(fmin/fs)-x0:int(fmax/fs)-x0] # Selected frequency range
        pp = power[int(fmin/fs)-x0:int(fmax/fs)-x0] # Power range

        # Reshape the power into n_rows of n_columns
        #  When modulus ~ zero, deltanu divides into frequency without remainder
        mod_zeros = find_peaks( -1.0*(ff % deltanu) )[0]

        # The bottom left corner of the plot is the lowest frequency that
        # divides into deltanu with almost zero remainder
        start = mod_zeros[0]

        # The top left corner of the plot is the highest frequency that
        # divides into deltanu with almost zero remainder.  This index is the
        # approximate end, because we fix an integer number of rows and columns
        approx_end = mod_zeros[-1]

        # The number of rows is the number of times you can partition your
        #  frequency range into chunks of size deltanu, start and ending at
        #  frequencies that divide nearly evenly into deltanu
        n_rows = len(mod_zeros) - 1

        # The number of columns is the total number of frequency points divided
        #  by the number of rows, floor divided to the nearest integer value
        n_columns =  int( (approx_end - start) / n_rows )

        # The exact end point is therefore the ncolumns*nrows away from the start
        end = start + n_columns*n_rows

        ep = np.reshape(pp[start : end], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        # Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[start : end], (n_rows, n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        y_f = (ef[:,0])
        return ep, x_f, y_f

    def _make_echelle_elements(self, deltanu, cmap='viridis',
        minimum_frequency=None, maximum_frequency=None, smooth_filter_width=.1,
        scale='linear', plot_width=490, plot_height=340, title='Echelle'):
        """Helper function to make the elements of the echelle diagram for bokeh plotting.
        """
        if not hasattr(deltanu, 'unit'):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable

        ep, x_f, y_f = self._clean_echelle(deltanu=deltanu,
                                           minimum_frequency=minimum_frequency,
                                           maximum_frequency=maximum_frequency,
                                           smooth_filter_width=smooth_filter_width,
                                           scale=scale)

        fig = figure(plot_width=plot_width, plot_height=plot_height,
                     x_range=(0, 1), y_range=(y_f[0].value, y_f[-1].value),
                     title=title, tools='pan,box_zoom,reset,lasso_select',
                     toolbar_location="above",
                     border_fill_color="white",tooltips=self.env.TOOLTIPS)

        fig.yaxis.axis_label = r'Frequency [{}]'.format(freq.unit.to_string())
        fig.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(deltanu)

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi)/500
        color_mapper = LogColorMapper(palette=self.palette, low=lo, high=hi)

        fig.image(image=[ep.value], x=x_f[0].value, y=y_f[0].value,
                  dw=1, dh=y_f[-1].value,
                  color_mapper=color_mapper, name='img')

        stretch_slider = RangeSlider(start=vlo,
                                     end=vhi,
                                     step=vstep,
                                     title='',
                                     value=(lo, hi),
                                     orientation='vertical',
                                     width=10,
                                     height=230,
                                     direction='rtl',
                                     show_value=False,
                                     sizing_mode='fixed',
                                     name='stretch')

        def stretch_change_callback(attr, old, new):
            """TPF stretch slider callback."""
            fig.select('img')[0].glyph.color_mapper.high = new[1]
            fig.select('img')[0].glyph.color_mapper.low = new[0]

        stretch_slider.on_change('value', stretch_change_callback)
        return fig, stretch_slider


    def interact_echelle(self, notebook_url="localhost:8888", **kwargs):
        """Display an interactive Jupyter notebook widget showing an Echelle diagram.

        This feature only works inside an active Jupyter Notebook, and
        requires an optional dependency, ``bokeh`` (v1.0 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        Parameters
        ----------
        notebook_url : str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        """
        try:
            import bokeh
            if bokeh.__version__[0] == '0':
                warnings.warn("interact() requires Bokeh version 1.0 or later", LightkurveWarning)
        except ImportError:
            log.error("The interact() tool requires the `bokeh` Python package; "
                      "you can install bokeh using e.g. `conda install bokeh`.")
            return None

        maximum_frequency = kwargs.pop('maximum_frequency', self.periodogram.frequency.max().value)
        minimum_frequency = kwargs.pop('minimum_frequency', self.periodogram.frequency. min().value)

        if not hasattr(self, 'deltanu'):
            dnu = SeismologyQuantity(quantity=self.periodogram.frequency.max()/30,
                                     name='deltanu', method='echelle')
        else:
            dnu = self.deltanu
        self.deltanu=dnu
        def create_interact_ui():
            self.env.fig_tpfint, self.env.stretch_sliderint = self._make_echelle_elements(dnu,
                                              maximum_frequency=maximum_frequency,
                                              minimum_frequency=minimum_frequency,
                                              **kwargs)
            # maxdnu = self.periodogram.frequency.max().value/5
            maxdnu = self.env.maxdnu
            # Interactive slider widgets
            self.env.dnu_slider = Slider(start=0.01,
                                end=maxdnu,
                                value=dnu.value,
                                step=0.01,
                                title="Delta Nu",
                                width=290)
            self.env.r_button = Button(label=">", button_type="default", width=30)
            self.env.l_button = Button(label="<", button_type="default", width=30)
            self.env.rr_button = Button(label=">>", button_type="default", width=30)
            self.env.ll_button = Button(label="<<", button_type="default", width=30)

            def update(attr, old, new):
                """Callback to take action when dnu slider changes"""
                dnu = SeismologyQuantity(quantity=self.env.dnu_slider.value*u.microhertz,
                                         name='deltanu', method='echelle')
                ep, x_f, y_f = self._clean_echelle(deltanu=dnu,
                                               minimum_frequency=self.env.minimum_frequency,
                                               maximum_frequency=self.env.maximum_frequency,
                                               **kwargs)
                self.env.fig_tpfint.select('img')[0].data_source.data['image'] = [ep.value]
                #test
                # print('x===========',x_f[0].value)
                # print('y===========',y_f[0].value)
                # self.env.fig_tpfint.select('img')[0].data_source.data['x'] = x_f.value
                # self.env.fig_tpfint.select('img')[0].data_source.data['y'] = y_f.value


                self.env.fig_tpfint.xaxis.axis_label = r'Frequency / {:.3f} Mod. 1'.format(dnu)
                self.env.dnu_val=self.env.dnu_slider.value
                self.env.dnu_text.value=str(self.env.dnu_slider.value)

            def go_right_by_one_small():
                """Step forward in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value < self.env.maxdnu:
                    self.env.dnu_slider.value = existing_value + 0.1

            def go_left_by_one_small():
                """Step back in time by a single cadence"""
                existing_value = self.dnu_slider.value
                if existing_value > 0:
                    self.env.dnu_slider.value = existing_value - 0.1

            def go_right_by_one():
                """Step forward in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value < self.env.maxdnu:
                    self.env.dnu_slider.value = existing_value + 1

            def go_left_by_one():
                """Step back in time by a single cadence"""
                existing_value = self.env.dnu_slider.value
                if existing_value > 0:
                    self.env.dnu_slider.value = existing_value - 1

            self.env.dnu_slider.on_change('value', update)
            self.env.r_button.on_click(go_right_by_one_small)
            self.env.l_button.on_click(go_left_by_one_small)
            self.env.rr_button.on_click(go_right_by_one)
            self.env.ll_button.on_click(go_left_by_one)

            # widgets_and_figures = layout([fig_tpf, [Spacer(height=20), stretch_slider]],
            #                              [ll_button, Spacer(width=30), l_button,
            #                               Spacer(width=25), dnu_slider, Spacer(width=30),
            #                               r_button, Spacer(width=23), rr_button])
            #doc.add_root(widgets_and_figures)
        create_interact_ui()

    # def find_mesa_osc(self):
    #     import mesa_reader as mr
    #     import mesamanager
    #     from mesamanager import taskmanager
    #     parsec_0_22 = {
    #         "initial_mass": [2.04],
    #         "initial_z": [0.0236],
    #         "initial_y": [0.2904],
    #         "mixing_length_alpha" : [2.0],

    #     }
    #     #mnum,mmax=taskmanager.get_mnum(parsec_0_22)
    #     mnum=3
    #     age=6e8
    #     mass=2.04
    #     mass_text='mesa_initial_mass_'+str(mass)
    #     y_text='__initial_y_'+str(parsec_0_22['initial_y'][0])
    #     z_text='__initial_z_'+str(parsec_0_22['initial_z'][0])
    #     mlt_text='__mixing_length_alpha_'+str(parsec_0_22['mixing_length_alpha'][0])
    #     directory='/media/dinilbose/Extdisk/mesamanager_data/data/0'
    #     print(directory)
    #     file=directory+'/LOGS'
    #     osc_file=directory+'/oscillations_ad.csv'
    #     # logs_dir=mr.MesaLogDir(file)
    #     # history=pd.DataFrame(logs_dir.history_data.bulk_data)
    #     # history=mesamanager.read_mesa_history(mnumber=0)
    #     #
    #     # nearest_point=history.loc[(abs(history.star_age-age)).idxmin()]
    #     # real_age=history.star_age[(abs(history.star_age-age)).idxmin()]
    #     # model_nearest=int(nearest_point['model_number'])
    #     # prof_data=pd.DataFrame(logs_dir.profile_data(model_number=model_nearest).bulk_data)
    #     # profile_nearest=logs_dir.profile_with_model_number(model_nearest)
    #     # print('profile_nearest',profile_nearest)
    #     # osc=pd.read_csv(osc_file)
    #     profile_nearest=[-1]
    #     # data=pd.read_csv(osc_file)
    #     # data=data.query('profile==@profile_nearest & n_pg>=0')

    #     data=mesamanager.read_mesa_oscillation_summary(mnum=1558)
    #     # data=data.query('profile==@profile_nearest & n_pg>=0')
    #     data=data.query('star_age ==@age')

    #     print (data)
    #     return data


    # def find_mesa_osc_update(self):
    #     import mesamanager.taskmanager as taskmanager
    #     import mesamanager
    #     import mesa_reader as mr

    #     initial_y=float(self.env.int_select_y.value)
    #     initial_z=float(self.env.int_select_z.value)
    #     initial_mass=float(self.env.int_select_mass.value)
    #     mixing_length_alpha=float(self.env.int_select_alpha.value)
    #     max_age=float(self.env.int_select_age.value)
    #     run_dict = {
    #         "initial_mass": initial_mass,
    #         "initial_z": initial_z,
    #         "initial_y": initial_y,
    #         "mixing_length_alpha" : mixing_length_alpha,
    #         "max_age": max_age,

    #     }
    #     print(run_dict)

    #     mnum,mmax=taskmanager.get_mnum(run_dict)
    #     print('print mnumber',mnum)
    #     age=max_age
    #     directory='/media/dinilbose/Extdisk/mesamanager_data/'+'data/'+str(mnum)
    #     file=directory+'/LOGS'
    #     osc_file=directory+'/oscillations_ad.csv'
    #     # logs_dir=mr.MesaLogDir(file)
    #     # history=pd.DataFrame(logs_dir.history_data.bulk_data)

    #     # history=mesamanager.read_mesa_history(mnumber=mnum)

    #     # nearest_point=history.loc[(abs(history.star_age-age)).idxmin()]
    #     # nearest_point=history.loc[((history.star_age==age)).idxmin()]
    #     # nearest_point=history.query('star_age==@age')

    #     # real_age=nearest_point.star_age.values[0]
    #     # print('age found',real_age)
    #     # model_nearest=int(nearest_point['model_number'])
    #     # print('age found',real_age,model_nearest)

    #     # prof_data=pd.DataFrame(logs_dir.profile_data(model_number=model_nearest).bulk_data)
    #     # profile_nearest=logs_dir.profile_with_model_number(model_nearest)

    #     # prof_data=mesamanager.read_mesa_profileindex(mnumber=mnum)
    #     # profile_nearest=prof_data.query('model_number==@model_nearest').profile_number.values[0]


    #     # osc=pd.read_csv(osc_file)
    #     #print(self.env.comp_data)
    #     # print('real_age',real_age,'star_age',age)
    #     # print('mnumber',mnum)
    #     # data_q=self.env.comp_data.query('name!="mesa"&max_age==@real_age &mnum==@mnum')
    #     # nm=data_q.name.values
    #     # print(data_q[['mnum','initial_mass','initial_z','initial_y','max_age','name']])
    #     # pr_str=[a.split('profile')[-1].replace('_','') for a in nm]
    #     # profile_nearest=np.unique([int(w) for w in pr_str])
    #     # if len(profile_nearest)==0:
    #     #      profile_nearest=np.array([0])
    #     # profile_nearest=list(profile_nearest)
    #     # print('profile_nearest',profile_nearest)

    #     # data=pd.read_csv(osc_file)
    #     data=mesamanager.read_mesa_oscillation_summary(mnum=mnum)
    #     # data=data.query('profile==@profile_nearest & n_pg>=0')
    #     data=data.query('star_age ==@age')
    #     print('Update Current query',self.env.text_osc_query.value)
    #     if self.env.text_osc_query.value:

    #         data=data.query(self.env.text_osc_query.value).reset_index(drop=True)
    #     #print(data)
    #     return data



    # def update_osc(self,attr,old,new):
    #     minfreq=float(self.env.frequency_minimum_text1.value)
    #     maxfreq=float(self.env.frequency_maximum_text1.value)
    #     self.env.mesa_osc_data['freq']=self.env.mesa_osc_data['Re(freq)']
    #     data=self.env.mesa_osc_data.query('freq>=@minfreq & freq<=@maxfreq')
    #     dnu=self.env.mesa_int_slider.value

    #     mesa_osc_n=data.query('l==0')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l0.data = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq)).data

    #     mesa_osc_n=data.query('l==1')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l1.data = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq)).data

    #     mesa_osc_n=data.query('l==2')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l2.data = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq)).data


    #     if self.env.plot_mesa_osc.active[0]==0:
    #         mesa_osc_n=data.query('l==0 & freq<=273')
    #         freq = mesa_osc_n['Re(freq)'].values
    #         self.env.tb_oscillation_modell0.data = ColumnDataSource(data=dict(x=freq,y=freq)).data

    #         mesa_osc_n=data.query('l==1 & freq<=273')
    #         freq = mesa_osc_n['Re(freq)'].values
    #         self.env.tb_oscillation_modell1.data = ColumnDataSource(data=dict(x=freq,y=freq)).data

    #         mesa_osc_n=data.query('l==2 & freq<=273')
    #         freq = mesa_osc_n['Re(freq)'].values
    #         self.env.tb_oscillation_modell2.data = ColumnDataSource(data=dict(x=freq,y=freq)).data
    #     else:
    #         self.env.tb_oscillation_modell0.data = ColumnDataSource(data=dict(x=[],y=[])).data
    #         self.env.tb_oscillation_modell1.data = ColumnDataSource(data=dict(x=[],y=[])).data
    #         self.env.tb_oscillation_modell2.data = ColumnDataSource(data=dict(x=[],y=[])).data




    # def update_osc_source(self,attr,old,new):
    #     dnu=self.env.source_int_slider.value
    #     minfreq=float(self.env.frequency_minimum_text1.value)
    #     maxfreq=float(self.env.frequency_maximum_text1.value)

    #     freq_source=self.env.tb_periodogram_se_tb.to_df().query('x>=@minfreq & x<=@maxfreq').x.values
    #     self.env.tb_source_osc.data=ColumnDataSource(data=dict(freq_dnu=freq_source%dnu, freq=freq_source)).data


    # def update_osc_source_button(self):
    #     # print('update press')
    #     self.env.mesa_osc_data=self.find_mesa_osc_update()
    #     # print('data update')
    #     self.update_osc_source(0,0,0)
    #     self.update_osc(0,0,0)

    # def mesa_interactive_plot(self,data):
    #     dnu=40
    #     self.env.fig_mesa_int= figure(plot_width=800, plot_height=400)

    #     mesa_osc_n=data.query('l==0')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l0 = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq))
    #     self.env.fig_mesa_int.circle("freq_dnu", "freq", source=self.env.tb_mesa_osc_l0, size=10, color="navy", alpha=0.5)

    #     mesa_osc_n=data.query('l==1')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l1 = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq))
    #     self.env.fig_mesa_int.square("freq_dnu", "freq", source=self.env.tb_mesa_osc_l1, size=10, color="navy", alpha=0.5)

    #     mesa_osc_n=data.query('l==2')
    #     freq = mesa_osc_n['Re(freq)'].values
    #     self.env.tb_mesa_osc_l2 = ColumnDataSource(data=dict(freq_dnu=freq%dnu, freq=freq))
    #     self.env.fig_mesa_int.asterisk("freq_dnu", "freq", source=self.env.tb_mesa_osc_l2, size=10, color="navy", alpha=0.5)

    #     freq_source=self.env.tb_periodogram_se_tb.to_df().x.values
    #     print('freq',freq_source)
    #     self.env.tb_source_osc=ColumnDataSource(data=dict(freq_dnu=freq_source%dnu, freq=freq_source))
    #     self.env.fig_mesa_int.circle("freq_dnu", "freq", source=self.env.tb_source_osc, size=10, color="red", alpha=0.5)



    # def mesa_interactive(self):


    #     # self.env.frequency_minimum_text1.on_change('value',self.update_osc_source)
    #     # self.env.frequency_maximum_text1.on_change('value',self.update_osc_source)


    #     self.env.mesa_osc_data=self.find_mesa_osc()
    #     dnu=48
    #     maxdnu=80
    #     self.mesa_interactive_plot(self.env.mesa_osc_data)
    #     self.env.mesa_int_slider = Slider(start=0.01,
    #                         end=maxdnu,
    #                         value=dnu,
    #                         step=0.01,
    #                         title="Gyre Delta Nu",
    #                         width=290)


    #     self.env.source_int_slider = Slider(start=0.01,
    #                         end=maxdnu,
    #                         value=dnu,
    #                         step=0.01,
    #                         title="Source Delta Nu",
    #                         width=290)






    #     self.env.mesa_int_slider.on_change('value', self.update_osc)
    #     self.env.source_int_slider.on_change('value', self.update_osc_source)
    #     self.env.tb_periodogram_se_tb.on_change('data',self.update_osc_source)



    #     self.env.frequency_minimum_text1=TextInput(value=str(0), title="F_min",width=150)
    #     self.env.frequency_maximum_text1=TextInput(value=str(1500), title="F_max",width=150)
    #     self.env.frequency_maxdnu_text1=TextInput(value=str(30), title="Maxdnu",width=150)
    #     self.env.update_int_source_button = Button(label="Update", button_type="success",width=150)

    #     self.env.update_int_source_button.on_click(self.update_osc_source_button)


    #     self.env.comp_data=pd.read_csv('/home/dinilbose/mesamanager/Completed_task.csv',float_precision='high')
    #     self.env.comp_data=self.env.comp_data.query('name!="mesa"')
    #     mass_list=[str(i) for i in self.env.comp_data.initial_mass.sort_values().unique()]
    #     self.env.int_select_mass = Select(title='mass', options=mass_list, value=mass_list[0])

    #     y_list=[str(i) for i in self.env.comp_data.initial_y.unique()]
    #     self.env.int_select_y = Select(title='y', options=y_list, value=y_list[0])

    #     z_list=[str(i) for i in self.env.comp_data.initial_z.unique()]
    #     self.env.int_select_z = Select(title='z', options=z_list, value=z_list[0])

    #     alpha_list=[str(i) for i in self.env.comp_data.mixing_length_alpha.unique()]
    #     self.env.int_select_alpha = Select(title='Alpha', options=alpha_list, value=alpha_list[0])

    #     age_list=[str(i) for i in self.env.comp_data.max_age.unique()]
    #     self.env.int_select_age = Select(title='age', options=age_list, value=age_list[0])


    #     self.env.int_select_y.on_change('value',self.update_y_change)
    #     self.env.int_select_z.on_change('value',self.update_z_change)
    #     self.env.int_select_mass.on_change('value',self.update_mass_change)
    #     self.env.int_select_alpha.on_change('value',self.update_alpha_change)
    #     self.env.int_select_age.on_change('value',self.update_age_change)

    #     self.env.update_int_reload_button = Button(label="Reload", button_type="success",width=150)
    #     self.env.update_int_reload_button.on_click(self.update_option)


    # def update_option(self):
    #     print('Reloading OptionsS')
    #     self.env.comp_data=pd.read_csv('/home/dinilbose/mesamanager/Completed_task.csv',float_precision='high')

    #     self.env.comp_data=self.env.comp_data.query('name!="mesa"')
    #     data=self.env.comp_data

    #     self.env.int_select_z.options=[str(i) for i in data.initial_z.unique()]

    #     self.env.int_select_age.options=[str(i) for i in data.max_age.unique()]

    #     self.env.int_select_alpha.options=[str(i) for i in data.mixing_length_alpha.unique()]

    #     self.env.int_select_mass.options=[str(i) for i in data.initial_mass.sort_values().unique()]

    #     self.env.int_select_y.options=[str(i) for i in data.initial_y.unique()]




    # def update_y_change(self,attr,old,new):
    #     if self.env.interactive_file_control==-1:
    #         self.env.interactive_file_control=0

    #         value=self.env.int_select_y.value
    #         data=self.env.comp_data.query('initial_y==@value')

    #         z=self.env.int_select_z.value
    #         if float(z) not in data.initial_z.to_list():
    #             self.env.int_select_z.value=str(data.initial_z.to_list()[0])
    #             self.env.int_select_z.options=[str(i) for i in data.initial_z.unique()]

    #         age=self.env.int_select_age.value
    #         if float(age) not in data.max_age.to_list():
    #             self.env.int_select_age.value=str(data.max_age.to_list()[0])
    #             self.env.int_select_age.options=[str(i) for i in data.max_age.unique()]

    #         alpha=self.env.int_select_alpha.value
    #         if float(alpha) not in data.mixing_length_alpha.to_list():
    #             print('y Alpha update')
    #             self.env.int_select_alpha.value=str(data.mixing_length_alpha.to_list()[0])
    #             self.env.int_select_alpha.options=[str(i) for i in data.mixing_length_alpha.unique()]

    #         mass=self.env.int_select_mass.value
    #         if float(mass) not in data.initial_mass.sort_values().to_list():
    #             self.env.int_select_mass.value=str(data.initial_mass.sort_values().to_list()[0])
    #             self.env.int_select_mass.options=[str(i) for i in data.initial_mass.sort_values().unique()]

    #         data=self.find_mesa_osc_update()
    #         self.env.mesa_osc_data=data
    #         self.update_osc(0,0,0)


    # def update_z_change(self,attr,old,new):
    #     if self.env.interactive_file_control==-1:
    #         self.env.interactive_file_control=0

    #         value=self.env.int_select_z.value
    #         data=self.env.comp_data.query('initial_z==@value')

    #         y=self.env.int_select_y.value
    #         if float(y) not in data.initial_y.to_list():
    #             self.env.int_select_y.value=str(data.initial_y.to_list()[0])
    #             self.env.int_select_y.options=[str(i) for i in data.initial_y.unique()]

    #         age=self.env.int_select_age.value
    #         if float(age) not in data.max_age.to_list():
    #             self.env.int_select_age.value=str(data.max_age.to_list()[0])
    #             self.env.int_select_age.options=[str(i) for i in data.max_age.unique()]

    #         alpha=self.env.int_select_alpha.value
    #         if float(alpha) not in data.mixing_length_alpha.to_list():
    #             self.env.int_select_alpha.value=str(data.mixing_length_alpha.to_list()[0])
    #             self.env.int_select_alpha.options=[str(i) for i in data.mixing_length_alpha.unique()]

    #         mass=self.env.int_select_mass.value
    #         if float(mass) not in data.initial_mass.sort_values().to_list():
    #             self.env.int_select_mass.value=str(data.initial_mass.sort_values().to_list()[0])
    #             self.env.int_select_mass.options=[str(i) for i in data.initial_mass.sort_values().unique()]
    #             print('z mass not found',mass,data.initial_mass.sort_values().to_list())

    #         data=self.find_mesa_osc_update()
    #         self.env.mesa_osc_data=data
    #         self.update_osc(0,0,0)


    # def update_mass_change(self,attr,old,new):
    #     if self.env.interactive_file_control==-1:
    #         self.env.interactive_file_control=0

    #         value=self.env.int_select_mass.value
    #         data=self.env.comp_data.query('initial_mass==@value')

    #         y=self.env.int_select_y.value
    #         if float(y) not in data.initial_y.to_list():
    #             self.env.int_select_y.value=str(data.initial_y.to_list()[0])
    #             self.env.int_select_y.options=[str(i) for i in data.initial_y.unique()]

    #         age=self.env.int_select_age.value
    #         if float(age) not in data.max_age.to_list():
    #             self.env.int_select_age.value=str(data.max_age.to_list()[0])
    #             self.env.int_select_age.options=[str(i) for i in data.max_age.unique()]

    #         alpha=self.env.int_select_alpha.value
    #         if float(alpha) not in data.mixing_length_alpha.to_list():
    #             self.env.int_select_alpha.value=str(data.mixing_length_alpha.to_list()[0])
    #             self.env.int_select_alpha.options=[str(i) for i in data.mixing_length_alpha.unique()]

    #         z=self.env.int_select_z.value
    #         if float(z) not in data.initial_z.to_list():
    #             self.env.int_select_z.value=str(data.initial_z.to_list()[0])
    #             self.env.int_select_z.options=[str(i) for i in data.initial_z.unique()]

    #         data=self.find_mesa_osc_update()
    #         self.env.mesa_osc_data=data
    #         self.update_osc(0,0,0)


    # def update_age_change(self,attr,old,new):
    #     if self.env.interactive_file_control==-1:
    #         self.env.interactive_file_control=0


    #         value=self.env.int_select_age.value
    #         data=self.env.comp_data.query('max_age==@value')

    #         y=self.env.int_select_y.value
    #         if float(y) not in data.initial_y.to_list():
    #             self.env.int_select_y.value=str(data.initial_y.to_list()[0])
    #             self.env.int_select_y.options=[str(i) for i in data.initial_y.unique()]

    #         alpha=self.env.int_select_alpha.value
    #         if float(alpha) not in data.mixing_length_alpha.to_list():
    #             self.env.int_select_alpha.value=str(data.mixing_length_alpha.to_list()[0])
    #             self.env.int_select_alpha.options=[str(i) for i in data.mixing_length_alpha.unique()]

    #         z=self.env.int_select_z.value
    #         if float(z) not in data.initial_z.to_list():
    #             self.env.int_select_z.value=str(data.initial_z.to_list()[0])
    #             self.env.int_select_z.options=[str(i) for i in data.initial_z.unique()]

    #         mass=self.env.int_select_mass.value
    #         if float(mass) not in data.initial_mass.sort_values().to_list():
    #             self.env.int_select_mass.value=str(data.initial_mass.sort_values().to_list()[0])
    #             self.env.int_select_mass.options=[str(i) for i in data.initial_mass.sort_values().unique()]

    #         data=self.find_mesa_osc_update()
    #         self.env.mesa_osc_data=data
    #         self.update_osc(0,0,0)

    # def update_alpha_change(self,attr,old,new):
    #     if self.env.interactive_file_control==-1:
    #         self.env.interactive_file_control=0

    #         print('Alpha Change')

    #         value=self.env.int_select_alpha.value
    #         data=self.env.comp_data.query('mixing_length_alpha==@value')

    #         y=self.env.int_select_y.value
    #         if float(y) not in data.initial_y.to_list():
    #             self.env.int_select_y.value=str(data.initial_y.to_list()[0])
    #             self.env.int_select_y.options=[str(i) for i in data.initial_y.unique()]

    #         z=self.env.int_select_z.value
    #         if float(z) not in data.initial_z.to_list():
    #             self.env.int_select_z.value=str(data.initial_z.to_list()[0])
    #             self.env.int_select_z.options=[str(i) for i in data.initial_z.unique()]

    #         mass=self.env.int_select_mass.value
    #         if float(mass) not in data.initial_mass.sort_values().to_list():
    #             self.env.int_select_mass.value=str(data.initial_mass.sort_values().to_list()[0])
    #             self.env.int_select_mass.options=[str(i) for i in data.initial_mass.sort_values().unique()]
    #         else:
    #             print('mass not in list')
    #         age=self.env.int_select_age.value
    #         if float(age) not in data.max_age.to_list():
    #             self.env.int_select_age.value=str(data.max_age.to_list()[0])
    #             self.env.int_select_age.options=[str(i) for i in data.max_age.unique()]

    #         data=self.find_mesa_osc_update()
    #         self.env.mesa_osc_data=data
    #         self.update_osc(0,0,0)
